import flet as ft
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import time
import sys
import io
import warnings
import textwrap
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel

warnings.filterwarnings('ignore')

try:
    from ProtFlash.pretrain import load_prot_flash_small
    from ProtFlash.utils import batchConverter
except ImportError:
    print("Hint: ProtFlash module missing.")


# --- Models ---

class Block(nn.Module):
    def __init__(self, in_d, hidden_ds):
        super().__init__()
        layers = []
        prev = in_d
        for h in hidden_ds:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GrowNet:
    def __init__(self, in_dim, n_models, h_dims, device='cpu'):
        self.device = device
        self.estimators = []
        for _ in range(n_models):
            m = Block(in_dim, h_dims).to(device)
            m.eval()
            self.estimators.append(m)

    def predict(self, x):
        xt = torch.FloatTensor(x).to(self.device) if not torch.is_tensor(x) else x.to(self.device)
        out = torch.zeros(len(x), 2, device=self.device)
        with torch.no_grad():
            for m in self.estimators:
                out += m(xt)
        return torch.softmax(out, dim=1).cpu().numpy()


# --- Backend ---

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ready = False
        if getattr(sys, 'frozen', False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(__file__)
        self.ankh_path = os.path.join(self.base_dir, "../ankh3")
        self.cfg = {}
        self.models = {}

    def load(self):
        if self.ready: return True, "OK"
        model_dir = os.path.join(self.base_dir, "../models/final_models")

        if not os.path.exists(model_dir): return False, f"Missing {model_dir}"

        try:
            self.cfg = joblib.load(os.path.join(model_dir, "config.pkl"))
            self.models['svm'] = joblib.load(os.path.join(model_dir, "svm.pkl"))
            self.models['rf'] = joblib.load(os.path.join(model_dir, "rf.pkl"))

            # Keys matched to Train.py
            gn_cfg = self.cfg['gn_cfg']
            dim = self.cfg['dim']

            gn_data = torch.load(os.path.join(model_dir, "grownet.pth"), map_location=self.device)
            gn = GrowNet(dim, gn_cfg['n_models'], gn_cfg['hiddens'], self.device)

            for i, st in enumerate(gn_data['models']):
                gn.estimators[i].load_state_dict(st)
            self.models['grownet'] = gn

            self.models['prot'] = load_prot_flash_small().to(self.device).eval()

            if not os.path.exists(self.ankh_path): return False, "Ankh path err"
            self.models['tok'] = T5Tokenizer.from_pretrained(self.ankh_path, local_files_only=True)
            self.models['ankh'] = T5EncoderModel.from_pretrained(
                self.ankh_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                local_files_only=True
            ).to(self.device).eval()

            self.ready = True
            return True, "Loaded"
        except Exception as e:
            return False, str(e)

    def predict_batch(self, seqs):
        dev = self.device
        preps = self.cfg['preps']
        bs = 32
        probs_list = []

        for i in range(0, len(seqs), bs):
            batch = seqs[i:i + bs]

            # ProtFlash
            p_feats = []
            tmp_data = [(f"p{k}", s[:512]) for k, s in enumerate(batch)]
            try:
                _, toks, lens = batchConverter(tmp_data)
                toks, lens = toks.to(dev), lens.to(dev)
                with torch.no_grad():
                    emb = self.models['prot'](toks, lens)
                    for k, (_, s) in enumerate(tmp_data):
                        p_feats.append(emb[k, :len(s) + 1].mean(0).cpu().numpy())
            except:
                p_feats = [np.zeros(512) for _ in batch]

            # Ankh
            a_feats = []
            a_in = ["[NLU]" + s[:500] for s in batch]
            try:
                enc = self.models['tok'](a_in, return_tensors="pt", padding=True, truncation=True, max_length=512)
                enc = {k: v.to(dev) for k, v in enc.items()}
                with torch.no_grad():
                    out = self.models['ankh'](**enc)
                    mask = enc['attention_mask'].unsqueeze(-1).float()
                    feats = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                    a_feats.append(feats.cpu().numpy())
            except:
                a_feats = [np.zeros(1536) for _ in batch]

            p_arr = np.nan_to_num(np.vstack(p_feats))
            a_arr = np.nan_to_num(np.vstack(a_feats)) if a_feats else np.zeros((len(batch), 1536))
            if isinstance(a_arr, list): a_arr = np.vstack(a_arr)

            p_s = preps['protflash_scaler'].transform(p_arr)
            a_s = preps['ankh3_scaler'].transform(a_arr)

            feat_final = preps['final_scaler'].transform(
                np.concatenate([
                    preps['protflash_reducer'].transform(p_s),
                    preps['ankh3_reducer'].transform(a_s)
                ], axis=1)
            )

            w = self.cfg['weights']
            sc_gn = self.models['grownet'].predict(feat_final)[:, 1]
            sc_svm = self.models['svm'].predict_proba(feat_final)[:, 1]
            sc_rf = self.models['rf'].predict_proba(feat_final)[:, 1]

            probs_list.extend(w[0] * sc_gn + w[1] * sc_svm + w[2] * sc_rf)

        final_probs = np.array(probs_list)
        preds = (final_probs >= self.cfg['threshold']).astype(int)
        return preds, final_probs


# --- UI ---

def main(page: ft.Page):
    page.title = "ACPSynergy"
    page.window_width, page.window_height = 1300, 950
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30
    page.bgcolor = "#f0f4f8"
    page.scroll = ft.ScrollMode.AUTO
    page.fonts = {"Roboto": "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"}
    page.theme = ft.Theme(font_family="Roboto", color_scheme_seed="blue")

    handler = ModelHandler()
    state = {"ids": [], "seqs": [], "probs": [], "preds": []}

    def save_csv(e):
        if not state["ids"]: return
        df = pd.DataFrame({
            "Sequence_ID": state["ids"], "Sequence": state["seqs"],
            "Probability": state["probs"], "Prediction": ["ACP" if p == 1 else "Non-ACP" for p in state["preds"]]
        })
        fp_save.save_file(allowed_extensions=["csv"], file_name="acp_results.csv")
        page.data = df

    fp_save = ft.FilePicker(on_result=lambda e: (page.data.to_csv(e.path, index=False) if e.path else None))
    page.overlay.append(fp_save)

    btn_dl = ft.ElevatedButton(
        "Download Results", icon=ft.Icons.DOWNLOAD,
        style=ft.ButtonStyle(
            color="white", bgcolor="#2196f3", padding=ft.padding.symmetric(20, 15),
            shape=ft.RoundedRectangleBorder(radius=8), text_style=ft.TextStyle(size=14, weight="bold")
        ),
        on_click=save_csv
    )

    txt_input = ft.TextField(
        multiline=True, expand=True,
        filled=False, hint_text="", border_color="transparent",
        bgcolor="white", text_size=16, content_padding=20,
        text_style=ft.TextStyle(font_family="Consolas,monospace"),
    )

    input_container = ft.Container(
        content=txt_input, height=350, bgcolor="white",
        border=ft.border.all(1, "#bbdefb"), border_radius=4,
    )

    EX1 = textwrap.dedent("""\
            >Sample_Seq_001
            ATCKAECPTWDSVCINKKPCVACCKKAKFSDGHCSKILRRCLCTKEC
            >Sample_Seq_002
            FFGTALKIAANILPTAICKILKKC""")

    EX2 = textwrap.dedent("""\
            >Polybia-MP1
            IDWKKLLDAAKQIL
            >Lactoferricin_B
            FKCRRWQWRMKKLGAPSITCVRRAF
            >Human_Insulin_B_Chain
            FVNQHLCGSHLVEALYLVCGERGFFYTPKT
            >Human_Glucagon
            HSQGTFTSDYSKYLDSRRAQDFVQWLMNT""")

    EX3 = textwrap.dedent("""\
            >Test_Sequence_01
            GIGKFLHSAKKFGKAFVGEIMNS
            >Test_Sequence_02
            KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK
            >Test_Sequence_03
            GIGAVLKVLTTGLPALISWIKRKRQQ
            >Test_Sequence_04
            LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES
            >Test_Sequence_05
            ILPWKWPWWPWRR
            >Test_Sequence_06
            RLCRIVVIRVCR
            >Test_Sequence_07
            TRSSRAGLQFPVGRVHRLLRK
            >Test_Sequence_08
            ACYCRIPACIAGERRYGTCIYQGRLWAFCC""")

    def set_ex(e):
        txt_input.value = e.control.data
        txt_input.update()

    s_btn = ft.ButtonStyle(color="#546e7a", side=ft.BorderSide(1, "#cfd8dc"), padding=15)
    b1 = ft.OutlinedButton("Example 1", data=EX1, on_click=set_ex, style=s_btn)
    b2 = ft.OutlinedButton("Example 2", data=EX2, on_click=set_ex, style=s_btn)
    b3 = ft.OutlinedButton("Example 3", data=EX3, on_click=set_ex, style=s_btn)

    lbl_file = ft.Text("No file selected", size=14, color="grey")

    def load_file(e):
        if e.files:
            with open(e.files[0].path, 'r') as f:
                txt_input.value = f.read()
                txt_input.update()
                lbl_file.value = f"Loaded: {e.files[0].name}"
                lbl_file.update()

    fp_load = ft.FilePicker(on_result=load_file)
    page.overlay.append(fp_load)

    txt_prog = ft.Text("Initializing...", size=24, color="#37474f", weight="bold")
    card_prog = ft.Container(
        visible=False, bgcolor="white", border_radius=12, padding=ft.padding.symmetric(40, 20),
        shadow=ft.BoxShadow(1, 20, "#d0d9e6"), border=ft.border.all(1, "white"), width=float('inf'),
        content=ft.Row([ft.ProgressRing(width=28, height=28, stroke_width=3, color="#2196f3"), txt_prog],
                       alignment="center", spacing=20)
    )

    card_sum = ft.Container(visible=False, width=float('inf'))
    grid_res = ft.ResponsiveRow(spacing=20, run_spacing=20)

    def parse_seqs(raw):
        ids, seqs = [], []
        txt = raw.strip()
        if not txt: return ids, seqs
        if ">" not in txt:
            for i, line in enumerate(txt.split('\n')):
                if line.strip():
                    ids.append(f"Seq_{i + 1}")
                    seqs.append(line.strip().upper())
        else:
            try:
                for r in SeqIO.parse(io.StringIO(txt), "fasta"):
                    ids.append(str(r.id))
                    seqs.append(str(r.seq).upper())
            except:
                pass
        return ids, seqs

    def update_results(ids, seqs, preds, probs, t_cost):
        card_prog.visible = False
        n_tot = len(preds)
        n_acp = np.sum(preds)
        n_neg = n_tot - n_acp

        def get_badge(count, color):
            return ft.Container(
                content=ft.Text(str(count), color="white", size=16, weight="bold"),
                bgcolor=color, width=40, height=40, alignment=ft.alignment.center, border_radius=20
            )

        card_sum.content = ft.Container(
            content=ft.Column([
                ft.Row([ft.Text("Prediction Summary", size=28, color="#1e88e5", weight="bold"), btn_dl],
                       alignment="spaceBetween"),
                ft.Text(f"Analysis complete for {n_tot} sequences in {t_cost:.2f}s.", size=16, color="#78909c"),
                ft.Divider(color="#eeeeee", thickness=1, height=30),
                ft.Row([ft.Text("Predicted ACPs", size=18, color="#37474f", expand=True),
                        get_badge(n_acp, "#4caf50")], alignment="spaceBetween", height=50),
                ft.Divider(color="#eeeeee", thickness=1, height=10),
                ft.Row([ft.Text("Predicted Non-ACPs", size=18, color="#37474f", expand=True),
                        get_badge(n_neg, "#ef5350")], alignment="spaceBetween", height=50),
            ]),
            bgcolor="white", padding=40, border_radius=12, shadow=ft.BoxShadow(1, 15, "#d0d9e6")
        )
        card_sum.visible = True

        cards = []
        for i in range(n_tot):
            is_acp = preds[i] == 1
            color = "#388e3c" if is_acp else "#d32f2f"
            tag = "ACP" if is_acp else "NOT-ACP"
            pct = probs[i] * 100

            content = ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text(ids[i], weight="bold", size=18, color="#263238", font_family="serif",
                                overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                        ft.Container(content=ft.Text(tag, color="white", size=12, weight="bold", font_family="serif"),
                                     bgcolor=color, padding=ft.padding.symmetric(10, 4), border_radius=4)
                    ], alignment="spaceBetween"),
                    ft.Container(
                        content=ft.Text(seqs[i], font_family="Consolas", size=14, color="#C2185B", no_wrap=False),
                        bgcolor="#E1F5FE", padding=15, border_radius=6, width=float('inf'),
                        margin=ft.margin.symmetric(vertical=10)),
                    ft.Row([
                        ft.Text("Probability Score", size=16, weight="bold", color="#37474f", font_family="serif"),
                        ft.Text(f"{pct:.2f}%", size=16, color=color, weight="bold", font_family="serif")
                    ], alignment="spaceBetween"),
                    ft.ProgressBar(value=probs[i], color=color, bgcolor="#eceff1", height=12, border_radius=6)
                ]),
                padding=25, bgcolor="white", border_radius=10, border=ft.border.all(1, "#eeeeee"),
                shadow=ft.BoxShadow(0, 5, "#eeeeee")
            )
            cards.append(ft.Column([content], col={"md": 6}))

        grid_res.controls = cards
        page.update()

    def run(e):
        ids, seqs = parse_seqs(txt_input.value)
        if not seqs:
            txt_input.error_text = "Please enter valid sequences"
            txt_input.update()
            return

        txt_input.value = ""
        txt_input.error_text = None
        txt_input.update()
        grid_res.controls.clear()
        card_sum.visible = False

        card_prog.visible = True
        txt_prog.value = "Initializing Models..." if not handler.ready else f"Processing {len(seqs)} sequences..."
        page.update()

        t0 = time.time()
        if not handler.ready:
            ok, msg = handler.load()
            if not ok:
                txt_prog.value, txt_prog.color = f"Error: {msg}", "red"
                page.update()
                return
            txt_prog.value = f"Processing {len(seqs)} sequences..."
            page.update()

        try:
            preds, probs = handler.predict_batch(seqs)
            state.update({"ids": ids, "seqs": seqs, "probs": probs, "preds": preds})
            update_results(ids, seqs, preds, probs, time.time() - t0)
        except Exception as ex:
            import traceback;
            traceback.print_exc()
            txt_prog.value, txt_prog.color = "Error Occurred", "red"
            page.update()

    sec_input = ft.Container(
        content=ft.Column([
            ft.Text("Input Sequences (FASTA)", weight="bold", size=18, color="#37474f"),
            ft.Container(height=10),
            input_container,
            ft.Container(height=15),
            ft.Row([
                ft.Row([b1, b2, b3], spacing=10, wrap=True),
                ft.OutlinedButton("Upload File", icon=ft.Icons.UPLOAD_FILE, style=s_btn,
                                  on_click=lambda _: fp_load.pick_files())
            ], alignment="spaceBetween"),
            lbl_file
        ]),
        bgcolor="white", padding=30, border_radius=12, shadow=ft.BoxShadow(1, 15, "#d0d9e6"), expand=6
    )

    desc_txt = (
        "ACPSynergy constructs a multi-level feature extraction framework by integrating "
        "ProtFlash and Ankh3 protein language models. It introduces GANs to enhance data "
        "diversity and utilizes the FusionMMRCore strategy for efficient feature sampling. "
        "Ultimately, it achieves high-precision anticancer peptide prediction via GrowNet, "
        "SVM, and Random Forest classifiers."
    )

    sec_img = ft.Container(
        content=ft.Column([
            ft.Text("Model Framework", weight="bold", size=18, color="#37474f"),
            ft.Container(height=10),
            ft.Image(src=os.path.join(os.path.dirname(__file__), "../famework.png"),
                     fit=ft.ImageFit.CONTAIN, border_radius=8, expand=True),
            ft.Container(height=10),
            ft.Container(
                content=ft.Text(desc_txt, size=14, color="#546e7a", text_align=ft.TextAlign.JUSTIFY,
                                style=ft.TextStyle(height=1.2)),
                padding=12, bgcolor="#fcfcfc", border_radius=8, border=ft.border.all(1, "#eceff1")
            )
        ], alignment="start"),
        bgcolor="white", padding=30, border_radius=12, shadow=ft.BoxShadow(1, 15, "#d0d9e6"), expand=4
    )

    page.add(ft.Column([
        ft.Container(
            content=ft.Column([
                ft.Text("ACPSynergy", weight="bold", size=32, color="#0d47a1"),
                ft.Text(
                    "Intelligent Anticancer Peptide Prediction System with Collaborative Dual-Modal Deep Semantic Encoding and Adaptive Balancing Strategy",
                    size=14, color="#546e7a")
            ], horizontal_alignment="center"),
            alignment=ft.alignment.center, padding=ft.padding.only(bottom=20)
        ),
        ft.Row([sec_input, sec_img], alignment="start", vertical_alignment="stretch", spacing=20, height=550),
        ft.Container(
            content=ft.ElevatedButton("Start Prediction", icon=ft.Icons.ROCKET_LAUNCH,
                                      style=ft.ButtonStyle(bgcolor="#2196f3", color="white", padding=25,
                                                           shape=ft.RoundedRectangleBorder(radius=10),
                                                           text_style=ft.TextStyle(size=18, weight="bold")),
                                      on_click=run
                                      ),
            padding=ft.padding.symmetric(vertical=25), alignment=ft.alignment.center
        ),
        card_prog, card_sum,
        ft.Container(height=20),
        ft.Text("Detailed Results", weight="bold", size=20, color="#37474f"),
        ft.Container(height=10),
        grid_res,
        ft.Container(height=50)
    ], expand=True))


if __name__ == "__main__":
    ft.app(target=main)