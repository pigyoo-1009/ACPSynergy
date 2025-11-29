import os
import sys
import warnings
import random
import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from transformers import T5Tokenizer, T5EncoderModel

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
sys.path.append(ROOT_DIR)

from ProtFlash.pretrain import load_prot_flash_small
from ProtFlash.utils import batchConverter

warnings.filterwarnings('ignore')

SEED = 42
ANKH_PATH = os.path.join(ROOT_DIR, "ankh3")
DATA_DIR = os.path.join(ROOT_DIR, "Dataset", "Train")
OUT_DIR = os.path.join(ROOT_DIR, "Features")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(SEED)


def load_fasta(path):
    seqs = []
    curr = ""
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if curr: seqs.append(curr)
                curr = ""
            else:
                curr += line
        if curr: seqs.append(curr)
    return seqs


def get_protflash(seqs, max_len=512, bs=32):
    print("ProtFlash encoding...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_prot_flash_small().to(device).eval()

    feats = []
    for i in range(0, len(seqs), bs):
        batch = seqs[i:i + bs]
        print(f"Batch {i // bs + 1}/{(len(seqs) - 1) // bs + 1}", end='\r')

        proc_seqs = [s[:max_len] if len(s) > max_len else s for s in batch]
        data = [(f"p{j}", s) for j, s in enumerate(proc_seqs)]

        _, batch_tok, lens = batchConverter(data)
        batch_tok = batch_tok.to(device)
        lens = lens.to(device)

        with torch.no_grad():
            emb = model(batch_tok, lens)
            for j, (_, s) in enumerate(data):
                # Mean pooling
                rep = emb[j, 0: len(s) + 1].mean(0)
                feats.append(rep.cpu().numpy())

        del batch_tok, lens, emb
        torch.cuda.empty_cache()

    return np.vstack(feats)


def get_ankh3(seqs, model_dir, max_len=512, bs=32):
    print(f"\nAnkh3 encoding ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tok = T5Tokenizer.from_pretrained(model_dir, local_files_only=True)
    model = T5EncoderModel.from_pretrained(model_dir, torch_dtype=torch.float16,
                                           device_map="auto", local_files_only=True)
    model.eval()

    feats = []
    for i in range(0, len(seqs), bs):
        batch = seqs[i:i + bs]
        print(f"Batch {i // bs + 1}/{(len(seqs) - 1) // bs + 1}", end='\r')

        # Add special token [NLU]
        proc = ["[NLU]" + (s[:max_len - 10] if len(s) > max_len - 10 else s) for s in batch]

        inputs = tok(proc, return_tensors="pt", padding=True, truncation=True,
                     max_length=max_len, add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            hid = out.last_hidden_state
            mask = inputs['attention_mask']

            mask_exp = mask.unsqueeze(-1).expand(hid.size()).float()
            masked_hid = hid * mask_exp
            sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            seq_feats = masked_hid.sum(dim=1) / sum_mask

            feats.append(seq_feats.cpu().numpy())

        del inputs, out, hid
        torch.cuda.empty_cache()

    return np.vstack(feats)


def process_features(seqs):
    p_raw = get_protflash(seqs)
    a_raw = get_ankh3(seqs, ANKH_PATH)

    print("\nProcessing & Reducing dimensions...")

    p_scl = StandardScaler()
    p_scaled = p_scl.fit_transform(p_raw)

    a_scl = StandardScaler()
    a_scaled = a_scl.fit_transform(a_raw)

    p_red = TruncatedSVD(n_components=256, random_state=SEED)
    p_low = p_red.fit_transform(p_scaled)

    a_red = GaussianRandomProjection(n_components=256, random_state=SEED)
    a_low = a_red.fit_transform(a_scaled)

    combined = np.concatenate([p_low, a_low], axis=1)

    final_scl = StandardScaler()
    final_X = final_scl.fit_transform(combined)

    tools = {
        'protflash_scaler': p_scl,
        'ankh3_scaler': a_scl,
        'protflash_reducer': p_red,
        'ankh3_reducer': a_red,
        'final_scaler': final_scl
    }

    return final_X, tools


def save_data(X, y, tools, prefix="features"):
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

    base = os.path.join(OUT_DIR, prefix)

    np.save(f"{base}_features.npy", X)
    np.save(f"{base}_labels.npy", y)
    joblib.dump(tools, f"{base}_preprocessors.pkl")

    with open(f"{base}.complete", 'w') as f: f.write("OK")
    print(f"Saved to {OUT_DIR} as '{prefix}_*'")


def main():
    if not torch.cuda.is_available():
        print("Warning: No GPU detected.")

    pos_path = os.path.join(DATA_DIR, 'TrainPos.fasta')
    neg_path = os.path.join(DATA_DIR, 'TrainNeg.fasta')

    print(f"Loading: {pos_path}")
    pos_seqs = load_fasta(pos_path)
    print(f"Loading: {neg_path}")
    neg_seqs = load_fasta(neg_path)

    all_seqs = pos_seqs + neg_seqs
    y = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))

    print(f"Total: {len(all_seqs)} (Pos: {len(pos_seqs)}, Neg: {len(neg_seqs)})")

    tgt_name = "features"
    if os.path.exists(os.path.join(OUT_DIR, f"{tgt_name}.complete")):
        if input(f"'{tgt_name}' exists. Overwrite? (y/n): ").lower() != 'y':
            return

    X, tools = process_features(all_seqs)
    save_data(X, y, tools, tgt_name)
    print("Done.")


if __name__ == "__main__":
    main()