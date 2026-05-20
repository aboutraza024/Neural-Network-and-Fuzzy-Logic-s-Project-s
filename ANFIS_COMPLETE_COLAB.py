"""
╔══════════════════════════════════════════════════════════════════╗
║   VEHICLE MPG CLASSIFICATION USING ANFIS                        ║
║   COMPLETE SINGLE-FILE VERSION — Paste directly into Colab      ║
║                                                                  ║
║   Authors : Muhammad Awab Sial, Syed Amber Ali Shah             ║
║   Course  : Neural Networks & Fuzzy Logic                       ║
║   Uni     : Bahria University, Islamabad                        ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO USE
──────────
1. Open Google Colab  →  File → New Notebook
2. Paste this entire file into a single code cell
3. Press Shift+Enter
4. Everything runs automatically — plots, rules, metrics, saved models

OR on Kaggle: New Notebook → paste into a code cell → Run All
"""

# ══════════════════════════════════════════════════════════════════
#  CELL 0 — IMPORTS & SETUP
# ══════════════════════════════════════════════════════════════════
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import product
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              classification_report, mean_squared_error)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE   = '.'   # output directory
print(f"✅  Imports done | Device: {DEVICE.upper()}")


# ══════════════════════════════════════════════════════════════════
#  CELL 1 — DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════

# ── 1a. Load ──────────────────────────────────────────────────────
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
       "/auto-mpg/auto-mpg.data")
col_names = ['mpg','cylinders','displacement','horsepower',
             'weight','acceleration','model_year','origin','car_name']
df = pd.read_csv(url, names=col_names, na_values='?', delim_whitespace=True)
print(f"✅  Dataset loaded: {df.shape}")

# ── 1b. Clean ─────────────────────────────────────────────────────
df = df.drop(columns=['car_name'])
df = df.dropna(subset=['horsepower'])
df['horsepower'] = pd.to_numeric(df['horsepower'])
print(f"   After cleaning: {df.shape}")

# ── 1c. Encode origin (one-hot) ───────────────────────────────────
df = pd.get_dummies(df, columns=['origin'], prefix='origin', drop_first=False)

# ── 1d. Create MPG class labels ───────────────────────────────────
def mpg_to_class(v):
    return 0 if v < 20 else (1 if v <= 30 else 2)
df['mpg_class'] = df['mpg'].apply(mpg_to_class)

FEATURE_COLS = ['cylinders','displacement','horsepower','weight',
                'acceleration','model_year','origin_1','origin_2','origin_3']

X_raw = df[FEATURE_COLS].values.astype(np.float32)
y_reg = df['mpg'].values.astype(np.float32)
y_cls = df['mpg_class'].values.astype(np.int64)

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw).astype(np.float32)

# ── 1e. Splits (70 / 15 / 15) ────────────────────────────────────
X_tr, X_tmp, yr_tr, yr_tmp, yc_tr, yc_tmp = train_test_split(
    X, y_reg, y_cls, test_size=0.30, stratify=y_cls, random_state=SEED)
X_val, X_te, yr_val, yr_te, yc_val, yc_te = train_test_split(
    X_tmp, yr_tmp, yc_tmp, test_size=0.50, stratify=yc_tmp, random_state=SEED)

print(f"   Train={len(X_tr)}  Val={len(X_val)}  Test={len(X_te)}")
print(f"   Class dist (train): {np.bincount(yc_tr)}")

# ── 1f. EDA Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Auto-MPG Dataset — EDA', fontsize=16, fontweight='bold')

ax = axes[0,0]
ax.hist(df['mpg'], bins=25, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(20, color='orange', linestyle='--', lw=1.8, label='Low/Med (20)')
ax.axvline(30, color='red',    linestyle='--', lw=1.8, label='Med/High (30)')
ax.set_title('MPG Distribution'); ax.set_xlabel('MPG'); ax.legend(fontsize=8)

ax = axes[0,1]
cont = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year']
corr = df[cont].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax, linewidths=0.5, annot_kws={'size':7},
            mask=np.triu(np.ones_like(corr, dtype=bool)))
ax.set_title('Feature Correlation')

ax = axes[1,0]
colors_map = {0:'tomato', 1:'gold', 2:'mediumseagreen'}
labels_map  = {0:'Low (<20)', 1:'Medium (20-30)', 2:'High (>30)'}
for cls in [0,1,2]:
    m = df['mpg_class']==cls
    ax.scatter(df.loc[m,'weight'], df.loc[m,'mpg'],
               c=colors_map[cls], label=labels_map[cls], alpha=0.5, s=16)
ax.set_title('Weight vs MPG'); ax.set_xlabel('Weight'); ax.set_ylabel('MPG')
ax.legend(fontsize=8)

ax = axes[1,1]
counts = df['mpg_class'].value_counts().sort_index()
bars = ax.bar([labels_map[i] for i in counts.index], counts.values,
              color=['tomato','gold','mediumseagreen'], edgecolor='white')
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            str(v), ha='center', fontsize=10)
ax.set_title('Class Distribution'); ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig(f'{SAVE}/eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("📈  EDA saved.")


# ══════════════════════════════════════════════════════════════════
#  CELL 2 — ANFIS ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

class FuzzificationLayer(nn.Module):
    """Layer 1: Gaussian MFs  μ(x) = exp(-0.5*((x-c)/σ)²)"""
    def __init__(self, n_inputs, n_mf=2):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mf     = n_mf
        init_c = torch.linspace(0.15, 0.85, n_mf).repeat(n_inputs, 1)
        self.centers    = nn.Parameter(init_c)
        self.log_sigmas = nn.Parameter(torch.full((n_inputs, n_mf), np.log(0.3)))

    @property
    def sigmas(self):
        return torch.exp(self.log_sigmas).clamp(min=1e-4)

    def forward(self, x):                               # x: [B, n_in]
        x_e = x.unsqueeze(2)                            # [B, n_in, 1]
        mu  = torch.exp(-0.5 * ((x_e - self.centers) / self.sigmas) ** 2)
        return mu                                        # [B, n_in, n_mf]


class RuleStrengthLayer(nn.Module):
    """Layer 2: w_k = Π_i μ_i,mf(k,i)(x_i)"""
    def __init__(self, n_inputs, n_mf):
        super().__init__()
        combos = list(product(range(n_mf), repeat=n_inputs))
        self.n_rules = len(combos)
        self.register_buffer('rules_idx',
                             torch.tensor(combos, dtype=torch.long))   # [R, n_in]

    def forward(self, mu):                              # mu: [B, n_in, n_mf]
        B = mu.shape[0]
        gathered = torch.zeros(B, self.n_rules, self.n_inputs if hasattr(self,'n_inputs')
                               else self.rules_idx.shape[1], device=mu.device)
        n_in = self.rules_idx.shape[1]
        for inp in range(n_in):
            mf_sel = self.rules_idx[:, inp]             # [R]
            gathered[:, :, inp] = mu[:, inp, :][:, mf_sel]
        return gathered.prod(dim=2)                     # [B, R]


class NormLayer(nn.Module):
    """Layer 3: w̄_k = w_k / Σ w_j"""
    def forward(self, w):
        return w / w.sum(dim=1, keepdim=True).clamp(min=1e-9)


class ConsequentLayer(nn.Module):
    """Layer 4: f_k = p_k0 + Σ p_ki * x_i  (Sugeno linear)"""
    def __init__(self, n_rules, n_inputs):
        super().__init__()
        self.P = nn.Parameter(torch.randn(n_rules, n_inputs+1) * 0.01)

    def forward(self, x, w_norm):                       # x:[B,n_in]  w_norm:[B,R]
        x_aug = torch.cat([torch.ones(x.size(0),1,device=x.device), x], dim=1)
        f     = x_aug @ self.P.T                        # [B, R]
        return (w_norm * f).sum(dim=1)                  # [B]


class ANFIS(nn.Module):
    """Full 5-layer ANFIS (regression output)"""
    def __init__(self, n_inputs, n_mf=2):
        super().__init__()
        self.l1 = FuzzificationLayer(n_inputs, n_mf)
        self.l2 = RuleStrengthLayer(n_inputs, n_mf)
        self.l3 = NormLayer()
        self.l4 = ConsequentLayer(self.l2.n_rules, n_inputs)
        print(f"🧠  ANFIS | inputs={n_inputs} | MFs={n_mf} | rules={self.l2.n_rules} "
              f"| params={sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        mu     = self.l1(x)
        w      = self.l2(mu)
        w_norm = self.l3(w)
        return self.l4(x, w_norm)

    def firing_strengths(self, x):
        with torch.no_grad():
            return self.l3(self.l2(self.l1(x)))


class ANFISClassifier(nn.Module):
    """ANFIS + classification head (3 classes)"""
    def __init__(self, n_inputs, n_mf=2, n_classes=3):
        super().__init__()
        self.anfis = ANFIS(n_inputs, n_mf)
        n_rules = self.anfis.l2.n_rules
        self.head = nn.Sequential(
            nn.Linear(n_rules, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes))

    def forward(self, x):
        mu     = self.anfis.l1(x)
        w      = self.anfis.l2(mu)
        w_norm = self.anfis.l3(w)
        return self.head(w_norm)                        # logits [B, 3]

    def reg_forward(self, x):
        return self.anfis(x)


N_INPUTS = X_tr.shape[1]
N_MF     = 2

# Quick sanity check
_m = ANFISClassifier(N_INPUTS, N_MF)
_o = _m(torch.rand(4, N_INPUTS))
print(f"✅  Architecture check — output shape: {_o.shape}")
del _m, _o


# ══════════════════════════════════════════════════════════════════
#  CELL 3 — TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════

def make_loader(Xa, ya, bs=32, shuffle=True):
    xt = torch.tensor(Xa, dtype=torch.float32)
    yt = torch.tensor(ya, dtype=torch.long if ya.dtype in [np.int32,np.int64] else np.float32)
    return DataLoader(TensorDataset(xt, yt), batch_size=bs, shuffle=shuffle)

def train_cls_epoch(model, loader, opt, crit, dev):
    model.train()
    tl, correct, n = 0., 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tl += loss.item()*len(xb); correct += (model(xb).argmax(1)==yb).sum().item(); n += len(xb)
    return tl/n, correct/n

@torch.no_grad()
def eval_cls(model, loader, crit, dev):
    model.eval()
    tl, correct, n = 0., 0, 0
    preds, trues = [], []
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        lg = model(xb); loss = crit(lg, yb)
        tl += loss.item()*len(xb)
        p = lg.argmax(1); correct += (p==yb).sum().item(); n += len(xb)
        preds.extend(p.cpu().numpy()); trues.extend(yb.cpu().numpy())
    preds, trues = np.array(preds), np.array(trues)
    return tl/n, correct/n, f1_score(trues,preds,average='weighted'), preds, trues

def train_reg_epoch(model, loader, opt, crit, dev):
    model.train()
    tl, n = 0., 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); loss = crit(model.anfis(xb), yb)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        tl += loss.item()*len(xb); n += len(xb)
    return tl/n

@torch.no_grad()
def eval_reg(model, loader, crit, dev):
    model.eval()
    tl, n = 0., 0; ps, ts = [], []
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        p = model.anfis(xb); loss = crit(p, yb)
        tl += loss.item()*len(xb); n += len(xb)
        ps.extend(p.cpu().numpy()); ts.extend(yb.cpu().numpy())
    ps, ts = np.array(ps), np.array(ts)
    return tl/n, float(np.sqrt(mean_squared_error(ts,ps))), ps, ts

print("✅  Training utilities defined.")


# ══════════════════════════════════════════════════════════════════
#  CELL 4 — 5-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════

CV_EPOCHS = 60
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_accs = []
print("🔁  5-Fold Cross-Validation …")

# Class weights for imbalance
cw = torch.tensor(1.0 / np.bincount(yc_tr), dtype=torch.float32).to(DEVICE)
cw = cw / cw.sum() * 3

for fold, (ti, vi) in enumerate(skf.split(X_tr, yc_tr)):
    Xf_tr, Xf_va = X_tr[ti], X_tr[vi]
    yf_tr, yf_va = yc_tr[ti], yc_tr[vi]

    ld_tr = make_loader(Xf_tr, yf_tr)
    ld_va = make_loader(Xf_va, yf_va, shuffle=False)

    m  = ANFISClassifier(N_INPUTS, N_MF).to(DEVICE)
    cr = nn.CrossEntropyLoss(weight=cw)
    op = optim.Adam(m.parameters(), lr=0.005, weight_decay=1e-4)
    sc = optim.lr_scheduler.CosineAnnealingLR(op, T_max=CV_EPOCHS)

    best = 0.
    for ep in range(CV_EPOCHS):
        train_cls_epoch(m, ld_tr, op, cr, DEVICE)
        _, va, _, _, _ = eval_cls(m, ld_va, cr, DEVICE)
        sc.step()
        if va > best: best = va

    fold_accs.append(best)
    print(f"   Fold {fold+1}/5 → {best*100:.2f}%")

print(f"\n   Mean={np.mean(fold_accs)*100:.2f}%  Std=±{np.std(fold_accs)*100:.2f}%")

# Plot
fig, ax = plt.subplots(figsize=(7,4))
ax.bar([f'Fold {i+1}' for i in range(5)], [a*100 for a in fold_accs],
       color='steelblue', edgecolor='white')
ax.axhline(np.mean(fold_accs)*100, color='red', linestyle='--', lw=1.8,
           label=f'Mean = {np.mean(fold_accs)*100:.2f}%')
ax.set_title('5-Fold Cross-Validation Accuracy', fontweight='bold')
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,105); ax.legend(); ax.grid(axis='y',alpha=0.3)
for i, a in enumerate(fold_accs):
    ax.text(i, a*100+1, f'{a*100:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{SAVE}/cross_validation.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════
#  CELL 5 — FULL TRAINING
# ══════════════════════════════════════════════════════════════════

EPOCHS = 150

# ── loaders ───────────────────────────────────────────────────────
ld = {
    'tr_cls': make_loader(X_tr,  yc_tr),
    'va_cls': make_loader(X_val, yc_val, shuffle=False),
    'te_cls': make_loader(X_te,  yc_te,  shuffle=False),
    'tr_reg': make_loader(X_tr,  yr_tr),
    'va_reg': make_loader(X_val, yr_val, shuffle=False),
    'te_reg': make_loader(X_te,  yr_te,  shuffle=False),
}

# ── A) CLASSIFICATION ─────────────────────────────────────────────
print("═"*55 + "\n  CLASSIFICATION ANFIS\n" + "═"*55)

cls_model = ANFISClassifier(N_INPUTS, N_MF).to(DEVICE)
crit_cls  = nn.CrossEntropyLoss(weight=cw)
opt_cls   = optim.Adam(cls_model.parameters(), lr=0.005, weight_decay=1e-4)
sch_cls   = optim.lr_scheduler.CosineAnnealingLR(opt_cls, T_max=EPOCHS)

h_cls     = {'tr_loss':[], 'va_loss':[], 'tr_acc':[], 'va_acc':[]}
best_vacc = 0.; best_state_cls = None

for ep in range(1, EPOCHS+1):
    tl, ta = train_cls_epoch(cls_model, ld['tr_cls'], opt_cls, crit_cls, DEVICE)
    vl, va, vf1, _, _ = eval_cls(cls_model, ld['va_cls'], crit_cls, DEVICE)
    sch_cls.step()
    h_cls['tr_loss'].append(tl); h_cls['va_loss'].append(vl)
    h_cls['tr_acc'].append(ta);  h_cls['va_acc'].append(va)
    if va > best_vacc:
        best_vacc = va
        best_state_cls = {k:v.clone() for k,v in cls_model.state_dict().items()}
    if ep % 30 == 0 or ep == 1:
        print(f"  Ep {ep:>4d}/{EPOCHS} | tr_loss={tl:.4f} tr_acc={ta:.4f} | va_loss={vl:.4f} va_acc={va:.4f}")

cls_model.load_state_dict(best_state_cls)
_, te_acc, te_f1, te_preds_cls, te_true_cls = eval_cls(cls_model, ld['te_cls'], crit_cls, DEVICE)
print(f"\n🎯  Test  Accuracy={te_acc:.4f}  F1={te_f1:.4f}")
print(classification_report(te_true_cls, te_preds_cls,
      target_names=['Low (<20)','Medium (20-30)','High (>30)']))

torch.save(cls_model.state_dict(), f'{SAVE}/anfis_cls_best.pt')

# ── B) REGRESSION ─────────────────────────────────────────────────
print("═"*55 + "\n  REGRESSION ANFIS\n" + "═"*55)

# Re-use the same ANFISClassifier but train only the anfis backbone for regression
reg_model = ANFISClassifier(N_INPUTS, N_MF).to(DEVICE)
crit_reg  = nn.MSELoss()
opt_reg   = optim.Adam(reg_model.parameters(), lr=0.005, weight_decay=1e-4)
sch_reg   = optim.lr_scheduler.CosineAnnealingLR(opt_reg, T_max=EPOCHS)

h_reg     = {'tr_mse':[], 'va_rmse':[]}
best_rmse = 1e9; best_state_reg = None

for ep in range(1, EPOCHS+1):
    tl = train_reg_epoch(reg_model, ld['tr_reg'], opt_reg, crit_reg, DEVICE)
    _, vrmse, _, _ = eval_reg(reg_model, ld['va_reg'], crit_reg, DEVICE)
    sch_reg.step()
    h_reg['tr_mse'].append(tl); h_reg['va_rmse'].append(vrmse)
    if vrmse < best_rmse:
        best_rmse = vrmse
        best_state_reg = {k:v.clone() for k,v in reg_model.state_dict().items()}
    if ep % 30 == 0 or ep == 1:
        print(f"  Ep {ep:>4d}/{EPOCHS} | tr_MSE={tl:.4f} | va_RMSE={vrmse:.4f}")

reg_model.load_state_dict(best_state_reg)
_, te_rmse, te_preds_reg, te_true_reg = eval_reg(reg_model, ld['te_reg'], crit_reg, DEVICE)
te_mae = float(np.mean(np.abs(te_preds_reg - te_true_reg)))
print(f"\n🎯  Test  RMSE={te_rmse:.4f}  MAE={te_mae:.4f}")
torch.save(reg_model.state_dict(), f'{SAVE}/anfis_reg_best.pt')


# ══════════════════════════════════════════════════════════════════
#  CELL 6 — TRAINING RESULT PLOTS
# ══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16,12))
fig.suptitle('ANFIS Training Results — Vehicle MPG', fontsize=16, fontweight='bold')
gs  = gridspec.GridSpec(2,2, figure=fig, hspace=0.38, wspace=0.30)

# (1) Classification loss
ax = fig.add_subplot(gs[0,0])
ax.plot(h_cls['tr_loss'], label='Train', lw=1.5)
ax.plot(h_cls['va_loss'], label='Val', lw=1.5, ls='--')
ax.set_title('Classification Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('CE Loss')
ax.legend(); ax.grid(alpha=0.3)

# (2) Classification accuracy
ax = fig.add_subplot(gs[0,1])
ax.plot(h_cls['tr_acc'], label='Train', lw=1.5)
ax.plot(h_cls['va_acc'], label='Val', lw=1.5, ls='--')
ax.set_title('Classification Accuracy'); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.legend(); ax.grid(alpha=0.3)

# (3) Confusion matrix
ax = fig.add_subplot(gs[1,0])
cm = confusion_matrix(te_true_cls, te_preds_cls)
im = ax.imshow(cm, cmap='Blues')
cnames = ['Low','Medium','High']
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(cnames); ax.set_yticklabels(cnames)
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax)
ax.set_title('Test Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('True')

# (4) Regression scatter
ax = fig.add_subplot(gs[1,1])
ax.scatter(te_true_reg, te_preds_reg, alpha=0.55, s=20, color='steelblue')
lo = min(te_true_reg.min(), te_preds_reg.min())-1
hi = max(te_true_reg.max(), te_preds_reg.max())+1
ax.plot([lo,hi],[lo,hi],'r--',lw=1.5,label='Perfect fit')
ax.set_title('Regression — Predicted vs True MPG')
ax.set_xlabel('True MPG'); ax.set_ylabel('Predicted MPG')
ax.legend(); ax.grid(alpha=0.3)

plt.savefig(f'{SAVE}/training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("📈  Training results saved.")


# ══════════════════════════════════════════════════════════════════
#  CELL 7 — INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════

MF_LABELS = {
    'cylinders'   : ['Few',    'Many'],
    'displacement': ['Small',  'Large'],
    'horsepower'  : ['Low',    'High'],
    'weight'      : ['Light',  'Heavy'],
    'acceleration': ['Slow',   'Fast'],
    'model_year'  : ['Old',    'Recent'],
    'origin_1'    : ['Non-USA','USA'],
    'origin_2'    : ['Non-EUR','European'],
    'origin_3'    : ['Non-JPN','Japanese'],
}
CLASS_NAMES = ['Low (<20 MPG)','Medium (20–30 MPG)','High (>30 MPG)']

# ── 7a. Membership functions ─────────────────────────────────────
layer   = cls_model.anfis.l1
centers = layer.centers.detach().cpu().numpy()
sigmas  = layer.sigmas.detach().cpu().numpy()
n_in    = layer.n_inputs; n_mf = layer.n_mf

cols = 3; rows = int(np.ceil(n_in/cols))
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.5*rows))
axes = axes.flatten()
fig.suptitle('Learned Gaussian Membership Functions', fontsize=14, fontweight='bold')
x_line = np.linspace(0,1,300)
pal = ['steelblue','tomato','mediumseagreen']
for i, feat in enumerate(FEATURE_COLS):
    ax = axes[i]
    lbls = MF_LABELS.get(feat, [f'MF{j}' for j in range(n_mf)])
    for j in range(n_mf):
        c, s = centers[i,j], sigmas[i,j]
        mu   = np.exp(-0.5*((x_line-c)/s)**2)
        ax.plot(x_line, mu, color=pal[j%3], lw=2,
                label=lbls[j] if j<len(lbls) else f'MF{j}')
        ax.axvline(c, color=pal[j%3], lw=0.8, ls=':', alpha=0.7)
    ax.set_title(feat.replace('_',' ').title(), fontsize=10)
    ax.set_xlabel('Norm. Value', fontsize=8); ax.set_ylabel('μ(x)', fontsize=8)
    ax.set_ylim(-0.05,1.1); ax.legend(fontsize=7); ax.grid(alpha=0.25)
for k in range(n_in, len(axes)):
    axes[k].set_visible(False)
plt.tight_layout()
plt.savefig(f'{SAVE}/membership_functions.png', dpi=150, bbox_inches='tight')
plt.show()
print("📈  MF plot saved.")

# ── 7b. Top fuzzy rules ───────────────────────────────────────────
X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
cls_model.eval()
anfis   = cls_model.anfis
with torch.no_grad():
    mu     = anfis.l1(X_te_t)
    w      = anfis.l2(mu)
    w_norm = anfis.l3(w)

avg_str  = w_norm.mean(0).cpu().numpy()
rules_ix = anfis.l2.rules_idx.cpu().numpy()
top10    = np.argsort(avg_str)[::-1][:10]

print("\n" + "═"*65)
print("  TOP 10 FUZZY RULES (by average firing strength on test set)")
print("═"*65)

rule_rows = []
for rank, ri in enumerate(top10):
    mf_idx = rules_ix[ri]
    ant = []
    for inp_j, mf_j in enumerate(mf_idx):
        feat = FEATURE_COLS[inp_j]
        lbls = MF_LABELS.get(feat, [f'MF{k}' for k in range(n_mf)])
        lbl  = lbls[mf_j] if mf_j < len(lbls) else f'MF{mf_j}'
        ant.append(f"{feat.replace('_',' ').title()} is {lbl}")

    # Approximate consequent via head weights for this rule
    hw = cls_model.head[0].weight.detach().cpu().numpy()  # [64, R]
    hb = cls_model.head[0].bias.detach().cpu().numpy()
    h  = np.tanh(hw[:, ri])
    ow = cls_model.head[3].weight.detach().cpu().numpy()
    sc = ow @ h
    cls_i = sc.argmax()

    s_pct = avg_str[ri]*100
    rule_str = " AND ".join(ant[:4]) + ("…" if len(ant)>4 else "")
    print(f"\n  #{rank+1:02d} [{s_pct:.2f}%]  IF {rule_str}")
    print(f"       THEN MPG is → {CLASS_NAMES[cls_i]}")
    rule_rows.append({'Rank':rank+1,'Rule':rule_str,
                      'Predicted Class':CLASS_NAMES[cls_i],
                      'Firing Strength (%)':round(s_pct,3)})

pd.DataFrame(rule_rows).to_csv(f'{SAVE}/top_rules.csv', index=False)
print(f"\n📝  Rules saved.")

# ── 7c. Rule importance bar chart ────────────────────────────────
top20 = np.argsort(avg_str)[::-1][:20]
fig, ax = plt.subplots(figsize=(12,5))
bars = ax.bar(range(20), avg_str[top20]*100, color='steelblue', edgecolor='white')
ax.set_xticks(range(20)); ax.set_xticklabels([f'R{i+1}' for i in range(20)])
ax.set_title('Top 20 Rules — Normalised Firing Strength (%)', fontweight='bold')
ax.set_xlabel('Rule ID (ranked)'); ax.set_ylabel('Avg Strength (%)')
ax.grid(axis='y', alpha=0.3)
for bar, v in zip(bars, avg_str[top20]*100):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'{v:.2f}%', ha='center', va='bottom', fontsize=6.5, rotation=45)
plt.tight_layout()
plt.savefig(f'{SAVE}/rule_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 7d. Explain one sample ────────────────────────────────────────
sample   = X_te_t[0:1]
with torch.no_grad():
    mu_s   = anfis.l1(sample); w_s = anfis.l2(mu_s); wn_s = anfis.l3(w_s)
    logit  = cls_model.head(wn_s); pred_s = logit.argmax(1).item()

print("\n" + "═"*55)
print("  SINGLE VEHICLE EXPLANATION")
print("═"*55)
print(f"  True  : {CLASS_NAMES[yc_te[0]]}")
print(f"  Pred  : {CLASS_NAMES[pred_s]}")
print("\n  Input Membership Degrees:")
mu_np = mu_s.squeeze(0).cpu().numpy()
for i, feat in enumerate(FEATURE_COLS):
    lbls = MF_LABELS.get(feat, [f'MF{k}' for k in range(n_mf)])
    vals = [f"{lbls[j] if j<len(lbls) else f'MF{j}'}={mu_np[i,j]:.3f}" for j in range(n_mf)]
    print(f"    {feat.replace('_',' ').title():18s} → {',  '.join(vals)}")
wn_np = wn_s.squeeze(0).cpu().numpy()
top3  = np.argsort(wn_np)[::-1][:3]
print("\n  Top-3 Active Rules:")
for rank, r in enumerate(top3):
    mfs = rules_ix[r]
    ant = [f"{FEATURE_COLS[j].replace('_',' ').title()} is "
           f"{MF_LABELS.get(FEATURE_COLS[j],['MF0','MF1'])[mfs[j]]}"
           for j in range(min(4,len(mfs)))]
    print(f"    [{rank+1}] w̄={wn_np[r]:.4f} | IF {' AND '.join(ant)} …")
print("═"*55)


# ══════════════════════════════════════════════════════════════════
#  CELL 8 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  FINAL RESULTS SUMMARY")
print("═"*55)
summary_df = pd.DataFrame({
    'Metric': ['CV Accuracy (mean)', 'CV Accuracy (std)',
               'Test Accuracy',      'Test F1 (weighted)',
               'Test RMSE (MPG)',    'Test MAE  (MPG)'],
    'Value' : [
        f"{np.mean(fold_accs)*100:.2f} %",
        f"± {np.std(fold_accs)*100:.2f} %",
        f"{te_acc*100:.2f} %",
        f"{te_f1:.4f}",
        f"{te_rmse:.4f}",
        f"{te_mae:.4f}",
    ]
})
print(summary_df.to_string(index=False))
summary_df.to_csv(f'{SAVE}/final_summary.csv', index=False)
print(f"\n📁  All outputs saved to: {SAVE}/")
print("✅  PROJECT COMPLETE.")
