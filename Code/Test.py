import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import joblib
import warnings
import os
import sys
import random
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(curr_dir))

from ProtFlash.pretrain import load_prot_flash_small
from ProtFlash.utils import batchConverter

warnings.filterwarnings('ignore')
SEED = 42


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class WeakNet(nn.Module):
    """GrowNet弱学习器"""
    def __init__(self, dim, hiddens):
        super().__init__()
        layers = []
        prev = dim
        for h in hiddens:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class GrowNet:
    """GrowNet模型容器"""
    def __init__(self, dim, n_models=18, hiddens=[256, 128], lr=0.005,
                 epochs=15, bs=64):
        self.dim = dim
        self.n_models = n_models
        self.hiddens = hiddens
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []

    def predict_proba(self, X):
        X_t = torch.FloatTensor(X).to(self.dev)
        ens = torch.zeros(len(X), 2, device=self.dev)
        for m in self.models:
            m.eval()
            with torch.no_grad(): ens += m(X_t)
        return torch.softmax(ens, dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def read_fas(path):
    seqs, curr = [], ""
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if curr: seqs.append(curr); curr = ""
            else:
                curr += line
        if curr: seqs.append(curr)
    return seqs


def get_prot(seqs, max_len=512, bs=32):
    """提取ProtFlash特征"""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_prot_flash_small().to(dev).eval()
    feats = []

    for i in range(0, len(seqs), bs):
        batch = seqs[i:i + bs]
        proc = [s[:max_len] if len(s) > max_len else s for s in batch]
        data = [(f"p{j}", s) for j, s in enumerate(proc)]

        ids, toks, lens = batchConverter(data)
        with torch.no_grad():
            emb = model(toks.to(dev), lens.to(dev))
            for j, (_, s) in enumerate(data):
                feats.append(emb[j, 0: len(s) + 1].mean(0).cpu().numpy())
        torch.cuda.empty_cache()

    return np.nan_to_num(np.vstack(feats), nan=0.0)


def get_ankh(seqs, path=None, max_len=512, bs=32):
    """提取Ankh3特征"""
    if not path: path = os.path.join("..", "ankh3")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tok = T5Tokenizer.from_pretrained(path, local_files_only=True)
    mod = T5EncoderModel.from_pretrained(path, torch_dtype=torch.float16,
                                         device_map="auto", local_files_only=True).eval()
    feats = []

    for i in range(0, len(seqs), bs):
        batch = seqs[i:i + bs]
        proc = ["[NLU]" + (s[:max_len - 10] if len(s) > max_len - 10 else s) for s in batch]

        inp = tok(proc, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=True)
        inp = {k: v.to(dev) for k, v in inp.items()}

        with torch.no_grad():
            out = mod(**inp)
            hs = out.last_hidden_state
            mask = inp['attention_mask'].unsqueeze(-1).expand(hs.size()).float()
            feats.append((hs * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9))
            feats[-1] = feats[-1].cpu().numpy()
        torch.cuda.empty_cache()

    return np.nan_to_num(np.vstack(feats), nan=0.0)


def proc_feats(f_prot, f_ankh, preps):
    """特征处理"""
    p_scl = preps['protflash_scaler'].transform(f_prot)
    a_scl = preps['ankh3_scaler'].transform(f_ankh)

    p_red = preps['protflash_reducer'].transform(p_scl)
    a_red = preps['ankh3_reducer'].transform(a_scl)

    comb = np.concatenate([p_red, a_red], axis=1)
    return preps['final_scaler'].transform(comb)


def calc_met(yt, yp, yprob=None):
    """计算指标"""
    yt_n, yp_n = np.asarray(yt), np.asarray(yp)
    tn, fp, fn, tp = confusion_matrix(yt_n, yp_n, labels=[0, 1]).ravel()

    acc = accuracy_score(yt_n, yp_n)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / den if den != 0 else 0
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(yt_n, np.asarray(yprob)) if yprob is not None else 0
    return float(acc), float(mcc), float(sn), float(sp), float(auc)


def load_all(path):
    """加载模型和配置"""
    print(f"正在加载预测模型: {path}")
    cfg = joblib.load(os.path.join(path, "config.pkl"))
    svm = joblib.load(os.path.join(path, "svm.pkl"))
    rf = joblib.load(os.path.join(path, "rf.pkl"))

    gc = cfg['gn_cfg']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gn = GrowNet(
        dim=cfg['dim'],
        n_models=gc['n_models'],
        hiddens=gc['hiddens'],
        lr=gc['lr'],
        epochs=gc['epochs'],
        bs=gc['bs']
    )

    st = torch.load(os.path.join(path, "grownet.pth"), map_location=dev)
    for s in st['models']:
        m = WeakNet(gn.dim, gn.hiddens).to(dev)
        m.load_state_dict(s)
        gn.models.append(m)

    return gn, svm, rf, cfg


if __name__ == "__main__":
    seed_all(SEED)
    if not torch.cuda.is_available(): sys.exit("未检测到GPU，程序退出")

    mod_path = os.path.join("..", "models", "final_models")
    test_dir = os.path.join("..", "Dataset", "Test")
    ankh_dir = os.path.join("..", "ankh3")

    gn, svm, rf, cfg = load_all(mod_path)
    w, thr, preps = np.array(cfg['weights']), cfg['threshold'], cfg['preps']

    p_file = os.path.join(test_dir, 'Test135P.fasta')
    n_file = os.path.join(test_dir, 'Test135N.fasta')

    seqs = read_fas(p_file) + read_fas(n_file)
    y_test = np.array([1] * len(read_fas(p_file)) + [0] * len(read_fas(n_file)))

    print("\n正在提取序列特征...")
    fp = get_prot(seqs)
    fa = get_ankh(seqs, path=ankh_dir)
    X_test = proc_feats(fp, fa, preps)

    print("\n正在进行集成预测...")
    # GrowNet
    pb_gn = gn.predict_proba(X_test)[:, 1]
    pd_gn = gn.predict(X_test)
    # SVM
    pb_svm = svm.predict_proba(X_test)[:, 1]
    pd_svm = svm.predict(X_test)
    # RF
    pb_rf = rf.predict_proba(X_test)[:, 1]
    pd_rf = rf.predict(X_test)
    # Ensemble
    pb_ens = w[0] * pb_gn + w[1] * pb_svm + w[2] * pb_rf
    pd_ens = (pb_ens >= thr).astype(int)

    m_gn = calc_met(y_test, pd_gn, pb_gn)
    m_svm = calc_met(y_test, pd_svm, pb_svm)
    m_rf = calc_met(y_test, pd_rf, pb_rf)
    m_ens = calc_met(y_test, pd_ens, pb_ens)

    print(f"\n独立测试集评估:")
    print(
        f"{'GrowNet':<10}: ACC={m_gn[0]:.4f}, MCC={m_gn[1]:.4f}, SN={m_gn[2]:.4f}, SP={m_gn[3]:.4f}, AUC={m_gn[4]:.4f}")
    print(
        f"{'SVM':<10}: ACC={m_svm[0]:.4f}, MCC={m_svm[1]:.4f}, SN={m_svm[2]:.4f}, SP={m_svm[3]:.4f}, AUC={m_svm[4]:.4f}")
    print(f"{'RF':<10}: ACC={m_rf[0]:.4f}, MCC={m_rf[1]:.4f}, SN={m_rf[2]:.4f}, SP={m_rf[3]:.4f}, AUC={m_rf[4]:.4f}")
    print(
        f"{'Ensemble':<10}: ACC={m_ens[0]:.4f}, MCC={m_ens[1]:.4f}, SN={m_ens[2]:.4f}, SP={m_ens[3]:.4f}, AUC={m_ens[4]:.4f}")