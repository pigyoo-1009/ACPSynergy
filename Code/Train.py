import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings
import os
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import cupy as cp

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(curr_dir))

warnings.filterwarnings('ignore')
SEED = 42


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_all(SEED)


class GAN:
    """过采样"""
    def __init__(self, n_target, z_dim=128, g_dim=(256, 256), d_dim=(256, 256),
                 epochs=300, bs=32, lr=2e-4, seed=None):
        self.n_target = n_target
        self.z_dim = z_dim
        self.g_dim = g_dim
        self.d_dim = d_dim
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.seed = seed

    def fit(self, X, y):
        if self.seed: seed_all(self.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_list, label_list = [], []

        for lbl in np.unique(y):
            sub_X = X[y == lbl]
            n_orig = len(sub_X)
            n_tgt = self.n_target.get(lbl, n_orig)

            data_list.append(sub_X)
            label_list.extend([lbl] * n_orig)

            n_need = max(n_tgt - n_orig, 0)
            if n_need > 0 and len(sub_X) > 10:
                syn = self._gen(sub_X, n_need, device)
                data_list.append(syn)
                label_list.extend([lbl] * n_need)

        return np.vstack(data_list), np.array(label_list)

    def _gen(self, data, n_gen, device):
        dim = data.shape[1]

        class Gen(nn.Module):
            def __init__(self, z, out, hiddens):
                super().__init__()
                layers = []
                prev = z
                for h in hiddens:
                    layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
                    prev = h
                layers.extend([nn.Linear(prev, out), nn.Tanh()])
                self.net = nn.Sequential(*layers)

            def forward(self, x): return self.net(x)

        class Disc(nn.Module):
            def __init__(self, inp, hiddens):
                super().__init__()
                layers = []
                prev = inp
                for h in hiddens:
                    layers.extend([nn.Linear(prev, h), nn.LeakyReLU(0.2), nn.Dropout(0.5)])
                    prev = h
                layers.append(nn.Linear(prev, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x): return self.net(x)

        G = Gen(self.z_dim, dim, self.g_dim).to(device)
        D = Disc(dim, self.d_dim).to(device)
        opt_G = torch.optim.Adam(G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        opt_D = torch.optim.Adam(D.parameters(), lr=self.lr, betas=(0.5, 0.9))

        scl = MinMaxScaler((-1, 1))
        x_scl = torch.FloatTensor(scl.fit_transform(data)).to(device)

        G.train();
        D.train()
        for _ in range(self.epochs):
            loader = DataLoader(TensorDataset(x_scl), batch_size=self.bs, shuffle=True)
            for (batch,) in loader:
                cur_bs = batch.size(0)

                # Train D
                opt_D.zero_grad()
                real_out = D(batch)
                loss_real = nn.BCEWithLogitsLoss()(real_out, torch.ones(cur_bs, 1, device=device))

                z = torch.randn(cur_bs, self.z_dim, device=device)
                fake = G(z)
                fake_out = D(fake.detach())
                loss_fake = nn.BCEWithLogitsLoss()(fake_out, torch.zeros(cur_bs, 1, device=device))

                loss_d = (loss_real + loss_fake) / 2
                loss_d.backward()
                opt_D.step()

                # Train G
                opt_G.zero_grad()
                z = torch.randn(cur_bs, self.z_dim, device=device)
                fake = G(z)
                out = D(fake)
                loss_g = nn.BCEWithLogitsLoss()(out, torch.ones_like(out))
                loss_g.backward()
                opt_G.step()

        G.eval()
        with torch.no_grad():
            z = torch.randn(n_gen, self.z_dim, device=device)
            gen = G(z).cpu().numpy()
            return scl.inverse_transform(gen)


class FussionMMRCore:
    """下采样"""
    def __init__(self, size, sim_thr=0.8, seed=None):
        self.size = size
        self.sim_thr = sim_thr
        self.seed = seed

    def fit(self, X, y):
        if self.seed: np.random.seed(self.seed)

        pos = X[y == 1]
        neg = X[y == 0]

        if len(neg) > self.size:
            idx = self._select(neg, self.size)
            neg_sel = neg[idx]
        else:
            neg_sel = neg

        X_res = np.vstack([pos, neg_sel])
        y_res = np.concatenate([np.ones(len(pos)), np.zeros(len(neg_sel))])
        return X_res, y_res

    def _select(self, data, k):
        n = len(data)
        if k >= n: return np.arange(n)

        dim = data.shape[1]
        if dim >= 512:
            f1, f2 = data[:, :256], data[:, 256:512]
        else:
            mid = dim // 2
            f1, f2 = data[:, :mid], data[:, mid:]

        chunk = 200
        if n <= chunk:
            sim1 = cosine_similarity(f1)
            sim2 = cosine_similarity(f2)
        else:
            s_idx = np.random.choice(n, min(chunk, n), replace=False)
            s_f1, s_f2 = f1[s_idx], f2[s_idx]
            part_sim1 = cosine_similarity(f1, s_f1)
            part_sim2 = cosine_similarity(f2, s_f2)

            sim1 = np.zeros((n, n))
            sim2 = np.zeros((n, n))
            for i in range(n):
                sim1[i, s_idx] = part_sim1[i]
                sim2[i, s_idx] = part_sim2[i]

        sim = 0.6 * sim1 + 0.4 * sim2
        sel = []
        mask = np.ones(n, dtype=bool)

        first = np.argmax(np.mean(sim, axis=1))
        sel.append(first)
        mask[first] = False

        for _ in range(k - 1):
            rem = np.where(mask)[0]
            if len(rem) == 0: break

            best_sc = -np.inf
            best_id = None

            for i in rem:
                min_s = np.min(sim[i, sel]) if sel else 0
                others = rem[rem != i]
                rep = np.mean(sim[i, others]) if len(others) > 0 else 0

                score = 0.7 * (1 - min_s) + 0.3 * rep if min_s < self.sim_thr else rep
                if score > best_sc:
                    best_sc = score
                    best_id = i

            if best_id is not None:
                sel.append(best_id)
                mask[best_id] = False

        return np.array(sel)


class WeakNet(nn.Module):
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
    """Gradient Boosting Neural Networks"""
    def __init__(self, dim, n_models=18, hiddens=[256, 128], lr=0.005,
                 epochs=15, bs=64, seed=None):
        self.dim = dim
        self.n_models = n_models
        self.hiddens = hiddens
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        self.seed = seed
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []

    def fit(self, X, y):
        if self.seed: torch.manual_seed(self.seed)
        X_t = torch.FloatTensor(X).to(self.dev)
        y_t = torch.LongTensor(y).to(self.dev)
        ens_pred = torch.zeros(len(X), 2, device=self.dev)

        for i in range(self.n_models):
            res = torch.zeros_like(ens_pred)
            res[range(len(y)), y_t] = 1.0
            if i > 0: res -= torch.softmax(ens_pred, dim=1)

            m = WeakNet(self.dim, self.hiddens).to(self.dev)
            opt = torch.optim.Adam(m.parameters(), lr=self.lr)
            crit = nn.MSELoss()

            loader = DataLoader(TensorDataset(X_t, res), batch_size=self.bs, shuffle=True)
            m.train()
            for _ in range(self.epochs):
                for bx, br in loader:
                    opt.zero_grad()
                    loss = crit(m(bx), br)
                    loss.backward()
                    opt.step()

            m.eval()
            with torch.no_grad():
                ens_pred += m(X_t)
            self.models.append(m)
        return self

    def predict_proba(self, X):
        X_t = torch.FloatTensor(X).to(self.dev)
        ens = torch.zeros(len(X), 2, device=self.dev)
        for m in self.models:
            m.eval()
            with torch.no_grad(): ens += m(X_t)
        return torch.softmax(ens, dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def balance_data(X, y, seed):
    """数据平衡"""
    pos, neg = X[y == 1], X[y == 0]
    yp, yn = y[y == 1], y[y == 0]

    # GAN 过采样
    gan = GAN({1: 983}, epochs=300, seed=seed)
    xp_over, yp_over = gan.fit(pos, yp)

    # 核心集下采样
    if len(neg) > 983:
        sampler = FussionMMRCore(983, seed=seed)
        xn_down, yn_down = sampler.fit(neg, yn)
    else:
        xn_down, yn_down = neg, yn

    return np.vstack([xp_over, xn_down]), np.concatenate([yp_over, yn_down])


def calc_metrics(y_true, y_pred, y_prob=None):
    """计算指标"""
    yt = cp.asarray(y_true)
    yp = cp.asarray(y_pred)

    tp = cp.sum((yt == 1) & (yp == 1))
    tn = cp.sum((yt == 0) & (yp == 0))
    fp = cp.sum((yt == 0) & (yp == 1))
    fn = cp.sum((yt == 1) & (yp == 0))

    acc = (tp + tn) / (tp + tn + fp + fn)
    den = cp.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / den if den != 0 else 0
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0

    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    return float(acc), float(mcc), float(sn), float(sp), float(auc)


def get_data(name):
    """加载特征"""
    f_dir = os.path.join("..", "Features")
    feats = np.load(os.path.join(f_dir, f"{name}_features.npy"))
    lbls = np.load(os.path.join(f_dir, f"{name}_labels.npy"))
    preps = joblib.load(os.path.join(f_dir, f"{name}_preprocessors.pkl"))
    return feats, lbls, preps


def get_preds(X, y, dim, dev, fold):
    """加载Fold预测"""
    res = {};
    mets = {}
    path = os.path.join("..", "models", "cv_models")

    # GrowNet
    d = torch.load(os.path.join(path, f"grownet_fold{fold}.pth"))
    gn = GrowNet(d['config']['input_dim'], d['config']['num_models'],
                 d['config']['hidden_dims'], d['config']['lr'],
                 d['config']['epochs_per_model'], d['config']['batch_size'])
    gn.models = []
    for s in d['models']:
        m = WeakNet(gn.dim, gn.hiddens).to(dev)
        m.load_state_dict(s)
        gn.models.append(m)

    pb = gn.predict_proba(X)[:, 1]
    pd = gn.predict(X)
    res['grownet'] = pb
    mets['grownet'] = calc_metrics(y, pd, pb)

    # SVM
    svm = joblib.load(os.path.join(path, f"svm_fold{fold}.pkl"))
    pb = svm.predict_proba(X)[:, 1]
    pd = svm.predict(X)
    res['svm'] = pb
    mets['svm'] = calc_metrics(y, pd, pb)

    # RF
    rf = joblib.load(os.path.join(path, f"rf_fold{fold}.pkl"))
    pb = rf.predict_proba(X)[:, 1]
    pd = rf.predict(X)
    res['rf'] = pb
    mets['rf'] = calc_metrics(y, pd, pb)

    return res, mets


def eval_cv(X, y, dev, w, thr):
    """CV评估"""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    all_preds, all_lbls = [], []
    scores = {'acc': [], 'mcc': [], 'sn': [], 'sp': [], 'auc': []}

    print("\n开始10折交叉验证...")

    for fold, (t_idx, v_idx) in enumerate(skf.split(X, y), 1):
        X_v, y_v = X[v_idx], y[v_idx]

        preds, mets = get_preds(X_v, y_v, X.shape[1], dev, fold)
        all_preds.append(preds)
        all_lbls.append(y_v)

        print(f"\n第 {fold}/10 折")
        for m in ['grownet', 'svm', 'rf']:
            v = mets[m]
            print(f"  {m.upper():<10}: ACC={v[0]:.4f}, MCC={v[1]:.4f}, SN={v[2]:.4f}, SP={v[3]:.4f}, AUC={v[4]:.4f}")

    # 计算集成
    for i in range(10):
        d = all_preds[i]
        yt = all_lbls[i]
        prob = w[0] * d['grownet'] + w[1] * d['svm'] + w[2] * d['rf']
        pred = (prob >= thr).astype(int)

        a, m, s, p, u = calc_metrics(yt, pred, prob)
        scores['acc'].append(a);
        scores['mcc'].append(m)
        scores['sn'].append(s);
        scores['sp'].append(p);
        scores['auc'].append(u)

    print(f"\n10折交叉验证集成平均结果:")
    print(f"  ACC: {np.mean(scores['acc']):.4f} ± {np.std(scores['acc']):.4f}")
    print(f"  MCC: {np.mean(scores['mcc']):.4f} ± {np.std(scores['mcc']):.4f}")
    print(f"  SN:  {np.mean(scores['sn']):.4f} ± {np.std(scores['sn']):.4f}")
    print(f"  SP:  {np.mean(scores['sp']):.4f} ± {np.std(scores['sp']):.4f}")
    print(f"  AUC: {np.mean(scores['auc']):.4f} ± {np.std(scores['auc']):.4f}")

    return {'w': w, 'thr': thr, 'scores': scores}


def train_save(X, y, w, thr, dev, preps):
    """训练保存"""
    print(f"\n训练并保存最终模型")
    s_dir = os.path.join("..", "models", "final_models")
    os.makedirs(s_dir, exist_ok=True)

    seed_all(SEED)
    X_bal, y_bal = balance_data(X, y, SEED)

    # GrowNet
    gn_cfg = {'n_models': 15, 'hiddens': [512, 256], 'lr': 0.001, 'epochs': 10, 'bs': 128}
    seed_all(SEED);
    torch.cuda.empty_cache()
    gn = GrowNet(X.shape[1], **gn_cfg, seed=SEED)
    gn.fit(X_bal, y_bal)
    torch.save({
        'models': [m.state_dict() for m in gn.models],
        'config': {'input_dim': gn.dim, 'num_models': gn.n_models, 'hidden_dims': gn.hiddens,
                   'lr': gn.lr, 'epochs_per_model': gn.epochs, 'batch_size': gn.bs}
    }, os.path.join(s_dir, "grownet.pth"))
    torch.cuda.empty_cache()

    # SVM
    svm_cfg = {'kernel': 'poly', 'C': 72.61414516028825, 'gamma': 'scale', 'degree': 2, 'probability': True,
               'random_state': SEED}
    seed_all(SEED)
    svm = SVC(**svm_cfg).fit(X_bal, y_bal)
    joblib.dump(svm, os.path.join(s_dir, "svm.pkl"))

    # RF
    rf_cfg = {'n_estimators': 500, 'max_depth': 30, 'min_samples_split': 10, 'min_samples_leaf': 4,
              'max_features': 'sqrt', 'random_state': 42}
    seed_all(SEED)
    rf = RandomForestClassifier(**rf_cfg).fit(X_bal, y_bal)
    joblib.dump(rf, os.path.join(s_dir, "rf.pkl"))

    # Config
    joblib.dump({
        'models': ['grownet', 'svm', 'rf'], 'weights': w.tolist(), 'threshold': thr,
        'dim': X.shape[1], 'preps': preps, 'gn_cfg': gn_cfg, 'rf_cfg': rf_cfg, 'svm_cfg': svm_cfg
    }, os.path.join(s_dir, "config.pkl"))


def main():
    if not torch.cuda.is_available(): sys.exit("No GPU")
    device = torch.device('cuda')
    print(f"使用GPU: {torch.cuda.get_device_name()}")

    print("加载已保存的特征...")
    X, y, preps = get_data('features')
    print(f"特征加载完成,维度: {X.shape}")

    w = np.array([0.45, 0.4, 0.15])
    thr = 0.46

    res = eval_cv(X, y, device, w, thr)
    train_save(X, y, res['w'], res['thr'], device, preps)


if __name__ == "__main__":
    main()