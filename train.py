"""
================================================================================
  MEDITATION STATE CLASSIFIER — TRAINING SCRIPT (SVM v3)
================================================================================
  Input : meditation_levels_output.csv
  Output: meditation_svm_classifier.pkl, model_feature_names.pkl, scaler.pkl,
          training_report.txt, confusion_matrix.png

  CHANGES FROM v2
  ───────────────
  1. Replaced RandomForestClassifier with SVC (RBF kernel).
  2. Hyperparameter search via GridSearchCV (C, gamma).
  3. Removed feature importance plot (not available for SVM kernel).
  4. All other diagnostics, SMOTE, CV, and scaling logic preserved.

  19 FEATURES:
    mean_rel_{band}       x5   mean relative power across channels
    std_rel_{band}        x5   spatial std across channels
    ratio_theta_alpha     x1   theta/alpha
    ratio_alpha_beta      x1   alpha/beta
    ratio_delta_alpha     x1   delta/alpha
    ratio_theta_beta      x1   theta/beta
    ratio_delta_beta      x1   delta/beta
    ratio_gamma_alpha     x1   gamma/alpha
    frontal_alpha_asymm   x1   left frontal alpha - right frontal alpha
    frontal_theta_mean    x1   mean frontal theta
    occipital_alpha_mean  x1   mean posterior alpha
================================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

warnings.filterwarnings("ignore")

# --- 1. PATHS ---
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(SCRIPT_DIR, 'meditation_levels_output.csv')
OUTPUT_DIR    = SCRIPT_DIR

MODEL_PATH    = os.path.join(OUTPUT_DIR, 'meditation_svm_classifier.pkl')
FEATURES_PATH = os.path.join(OUTPUT_DIR, 'model_feature_names.pkl')
SCALER_PATH   = os.path.join(OUTPUT_DIR, 'scaler.pkl')
REPORT_PATH   = os.path.join(OUTPUT_DIR, 'training_report.txt')

# --- 2. CONSTANTS ---
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

LEVEL_NAMES = {
    0: "Level 0 — Baseline (Beta/Gamma dominant)",
    1: "Level 1 — Relaxed (Alpha dominant)",
    2: "Level 2 — Focused Attention (Theta dominant)",
    3: "Level 3 — Deep Meditation (Delta dominant)",
}

MUSE_CHANNEL_MAP = {
    'TP9':  ['TP7', 'T7', 'TP9', 'P7', 'P9'],
    'AF7':  ['AF3', 'F7', 'AF7', 'Fp1', 'F5', 'F3'],
    'AF8':  ['AF4', 'F8', 'AF8', 'Fp2', 'F6', 'F4'],
    'TP10': ['TP8', 'T8', 'TP10', 'P8', 'P10'],
}

FRONTAL_LEFT_CHANNELS  = ['AF3', 'F3', 'F7', 'AF7', 'Fp1', 'F5', 'FC1', 'FC3']
FRONTAL_RIGHT_CHANNELS = ['AF4', 'F4', 'F8', 'AF8', 'Fp2', 'F6', 'FC2', 'FC4']
FRONTAL_THETA_CHANNELS = ['AF3', 'AF4', 'AF7', 'AF8', 'F3', 'F4', 'FZ', 'FCZ',
                          'Fp1', 'Fp2', 'F7', 'F8']
POSTERIOR_ALPHA_CHANNELS = ['O1', 'O2', 'OZ', 'PO3', 'PO4', 'PO7', 'PO8',
                            'P7', 'P8', 'TP7', 'TP8', 'TP9', 'TP10', 'P9', 'P10']

# --- 3. LOAD DATA ---
print("=" * 70)
print("  MEDITATION CLASSIFIER — TRAINING (SVM v3)")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA] Loaded {len(df)} rows x {df.shape[1]} columns")
print(f"[DATA] Meditation level distribution:")
print(df['meditation_level'].value_counts().sort_index().to_string())

# --- 4. SUBJECT GROUPS ---
subject_col = 'subject' if 'subject' in df.columns else None
if subject_col:
    groups     = df[subject_col].values
    n_subjects = len(np.unique(groups))
    print(f"\n[DATA] Subjects found: {n_subjects}")
    print(f"[DATA] Rows per subject (min/mean/max): "
          f"{df.groupby(subject_col).size().min()} / "
          f"{df.groupby(subject_col).size().mean():.1f} / "
          f"{df.groupby(subject_col).size().max()}")
else:
    groups = None
    n_subjects = 0
    print("\n[WARN] No 'subject' column — cannot do subject-independent CV")

# --- 5. IDENTIFY BAND POWER COLUMNS ---
META_COLS = {'Condition', 'TRIGGER', 'M1', 'M2', 'group', 'subject',
             'time_window', 'filepath', 'cluster', 'meditation_level', 'level_name'}

rel_band_cols = {}
abs_band_cols = {}
muse_rel_band_cols = {}
muse_abs_band_cols = {}

all_muse_ch = set()
for ch_list in MUSE_CHANNEL_MAP.values():
    all_muse_ch.update([c.upper() for c in ch_list])

print()
for band in BANDS:
    rel_band_cols[band] = [c for c in df.columns
                           if f'_{band}/all' in c
                           and not any(m in c for m in META_COLS)]
    abs_band_cols[band] = [c for c in df.columns
                           if f'_{band}' in c and '/all' not in c
                           and not any(m in c for m in META_COLS)]

    muse_rel_band_cols[band] = [
        c for c in rel_band_cols[band]
        if any(ch.upper() in c.upper() for ch in all_muse_ch)
    ]
    muse_abs_band_cols[band] = [
        c for c in abs_band_cols[band]
        if any(ch.upper() in c.upper() for ch in all_muse_ch)
    ]

    print(f"  {band:7s}: {len(rel_band_cols[band]):3d} rel cols | "
          f"{len(muse_rel_band_cols[band]):3d} Muse-proximity rel cols")

muse_coverage = sum(1 for b in BANDS if muse_rel_band_cols[b] or muse_abs_band_cols[b])
if muse_coverage >= 4:
    use_rel = muse_rel_band_cols
    use_abs = muse_abs_band_cols
    channel_strategy = "Muse-proximity (reduced domain gap)"
    print(f"\n[STRATEGY] Using Muse-proximity channels ({muse_coverage}/5 bands covered)")
else:
    use_rel = rel_band_cols
    use_abs = abs_band_cols
    channel_strategy = "All channels (no Muse-proximity columns found)"
    print(f"\n[STRATEGY] Using all channels (Muse-proximity coverage: {muse_coverage}/5)")

all_power_cols = [c for b in BANDS for c in use_rel[b] + use_abs[b]]
df[all_power_cols] = df[all_power_cols].apply(pd.to_numeric, errors='coerce')
for c in all_power_cols:
    if df[c].isna().any():
        df[c].fillna(df[c].median(), inplace=True)

# --- 6. ENGINEER FEATURES ---
print("\n[FEATURES] Engineering 19 features...")
eps  = 1e-8
feat = pd.DataFrame(index=df.index)

for band in BANDS:
    if use_rel[band]:
        feat[f'mean_rel_{band}'] = df[use_rel[band]].mean(axis=1)
    elif use_abs[band]:
        total = sum(df[use_abs[b]].mean(axis=1) for b in BANDS) + eps
        feat[f'mean_rel_{band}'] = df[use_abs[band]].mean(axis=1) / total
    else:
        feat[f'mean_rel_{band}'] = 0.0
        print(f"  [WARN] No columns for {band} — set to 0")

for band in BANDS:
    if use_rel[band]:
        feat[f'std_rel_{band}'] = df[use_rel[band]].std(axis=1).fillna(0)
    elif use_abs[band]:
        total = sum(df[use_abs[b]].mean(axis=1) for b in BANDS) + eps
        feat[f'std_rel_{band}'] = (
            df[use_abs[band]].div(total, axis=0).std(axis=1).fillna(0)
        )
    else:
        feat[f'std_rel_{band}'] = 0.0

feat['ratio_theta_alpha'] = feat['mean_rel_Theta'] / (feat['mean_rel_Alpha'] + eps)
feat['ratio_alpha_beta']  = feat['mean_rel_Alpha'] / (feat['mean_rel_Beta']  + eps)
feat['ratio_delta_alpha'] = feat['mean_rel_Delta'] / (feat['mean_rel_Alpha'] + eps)
feat['ratio_theta_beta']  = feat['mean_rel_Theta'] / (feat['mean_rel_Beta']  + eps)
feat['ratio_delta_beta']  = feat['mean_rel_Delta'] / (feat['mean_rel_Beta']  + eps)
feat['ratio_gamma_alpha'] = feat['mean_rel_Gamma'] / (feat['mean_rel_Alpha'] + eps)

left_fa = [c for c in use_rel['Alpha']
           if any(ch.upper() in c.upper() for ch in FRONTAL_LEFT_CHANNELS)]
right_fa = [c for c in use_rel['Alpha']
            if any(ch.upper() in c.upper() for ch in FRONTAL_RIGHT_CHANNELS)]
feat['frontal_alpha_asymm'] = (
    df[left_fa].mean(axis=1) - df[right_fa].mean(axis=1)
    if left_fa and right_fa else pd.Series(0.0, index=df.index)
)

frontal_theta = [c for c in use_rel['Theta']
                 if any(ch.upper() in c.upper() for ch in FRONTAL_THETA_CHANNELS)]
feat['frontal_theta_mean'] = (df[frontal_theta].mean(axis=1)
                              if frontal_theta else feat['mean_rel_Theta'])

occ_alpha = [c for c in use_rel['Alpha']
             if any(ch.upper() in c.upper() for ch in POSTERIOR_ALPHA_CHANNELS)]
feat['occipital_alpha_mean'] = (df[occ_alpha].mean(axis=1)
                                if occ_alpha else feat['mean_rel_Alpha'])

feature_cols = list(feat.columns)
print(f"[FEATURES] Total: {len(feature_cols)}")
for i, f in enumerate(feature_cols):
    print(f"    [{i:02d}] {f}")

X = feat.values.astype(np.float64)
y = df['meditation_level'].values.astype(int)

bad = ~np.isfinite(X).all(axis=1)
if bad.any():
    X = X[~bad]; y = y[~bad]
    df = df[~bad].reset_index(drop=True)
    groups = groups[~bad] if groups is not None else None

valid = np.isin(y, [0, 1, 2, 3])
if not valid.all():
    X = X[valid]; y = y[valid]
    df = df[valid].reset_index(drop=True)
    groups = groups[valid] if groups is not None else None

print(f"\n[DATA] Final: {len(y)} samples")
print(f"[DATA] Class counts: "
      f"{{int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}} }")

# --- 6b. LABEL CIRCULARITY DIAGNOSTIC ---
print("\n[DIAGNOSTIC] Checking for label circularity...")

for d in [1, 2, 3]:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X, y)
    print(f"  Decision tree (depth={{d}}) train acc: {{accuracy_score(y, dt.predict(X)):.3f}}")

dt1 = DecisionTreeClassifier(max_depth=1, random_state=42)
dt1.fit(X, y)
dt1_acc = accuracy_score(y, dt1.predict(X))
label_circularity_warning = dt1_acc > 0.70

if label_circularity_warning:
    print(f"\n  ⚠️  Depth-1 tree gets {{dt1_acc:.1%}}. Labels are likely derived from")
    print("     the band-power features. Model learns a circular mapping.")
    print("     Results will look good on paper but won't generalise well to")
    print("     truly novel EEG data from different recording conditions.\n")
else:
    print(f"\n  ✓ No obvious circularity (depth-1 acc = {{dt1_acc:.1%}})\n")

# --- 7. SVM HYPERPARAMETER SEARCH ---
print("-" * 70)
print("[SVM] Tuning hyperparameters via GridSearchCV (RBF kernel)...")

SVM_PARAM_GRID = {
    'C':     [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001],
}

# Scale first for search
_sc_pre = StandardScaler()
X_pre   = _sc_pre.fit_transform(X)

gs = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    SVM_PARAM_GRID,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
)
gs.fit(X_pre, y)
best_C     = gs.best_params_['C']
best_gamma = gs.best_params_['gamma']

print(f"[SVM] Best params : C={{best_C}}, gamma={{best_gamma}}")
print(f"[SVM] Best CV acc : {{gs.best_score_:.3f}}")

SVM_PARAMS = dict(
    kernel='rbf',
    C=best_C,
    gamma=best_gamma,
    class_weight='balanced',
    probability=True,
    random_state=42,
)

# --- 8. CROSS-VALIDATION ---
print("-" * 70)
print("[CV] Subject-independent cross-validation\n")

if groups is not None and n_subjects >= 5:
    n_splits = min(5, n_subjects)
    cv       = GroupKFold(n_splits=n_splits)
    cv_label = f"Subject-Independent GroupKFold (n_splits={{n_splits}})" 

    fold_accs  = []
    all_y_true = []
    all_y_pred = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if HAS_SMOTE:
            min_count = min(np.bincount(y_tr)[np.bincount(y_tr) > 0])
            if min_count >= 6:
                smote = SMOTE(random_state=42,
                              k_neighbors=min(5, min_count - 1))
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        fold_clf = SVC(**SVM_PARAMS)
        fold_clf.fit(X_tr_s, y_tr)
        y_pred_fold = fold_clf.predict(X_te_s)

        acc = accuracy_score(y_te, y_pred_fold)
        fold_accs.append(acc)
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred_fold)

        subj_test = np.unique(groups[test_idx])
        dist = dict(zip(*np.unique(y_te, return_counts=True)))
        print(f"  Fold {{fold_i+1}}: subjects={{subj_test}}  n={{len(y_te)}}  "
              f"acc={{acc:.3f}}  dist={{dist}}")

    cv_scores  = np.array(fold_accs)
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
overall_cv_acc = accuracy_score(all_y_true, all_y_pred)

from scipy import stats as sp_stats
if len(cv_scores) >= 3:
    ci_low, ci_high = sp_stats.t.interval(
        0.95, df=len(cv_scores)-1,
        loc=np.mean(cv_scores), scale=sp_stats.sem(cv_scores))
else:
    ci_low, ci_high = cv_scores.min(), cv_scores.max()

print(f"\n[CV] Strategy    : {{cv_label}}")
print(f"[CV] Folds       : {{np.round(cv_scores, 3)}}")
print(f"[CV] Mean +/- Std: {{cv_scores.mean():.3f}} +/- {{cv_scores.std():.3f}}")
print(f"[CV] 95% CI      : [{{ci_low:.3f}}, {{ci_high:.3f}}]")
print(f"[CV] Overall     : {{overall_cv_acc:.3f}}")
if HAS_SMOTE:
    print(f"[CV] SMOTE       : applied per fold")

present_labels = sorted(np.unique(y))
cv_report = classification_report(
    all_y_true, all_y_pred,
    labels=present_labels,
    target_names=[LEVEL_NAMES[i] for i in present_labels]
)
print(f"\n{{cv_report}}\n")

for lvl in present_labels:
    n_test = (all_y_true == lvl).sum()
    if n_test < 20:
        print(f"  ⚠️  Class {{lvl}}: only {{n_test}} test samples — metric unreliable")

cm_cv = confusion_matrix(all_y_true, all_y_pred, labels=present_labels)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=cm_cv,
    display_labels=[f"L{{i}}" for i in present_labels]
).plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f"Confusion Matrix — Subject-Independent CV (acc={{overall_cv_acc:.3f}})",
             fontsize=10)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.close()
else:
    print("[CV] No subject column — StratifiedKFold (optimistic)")
    cv_label = "StratifiedKFold — WARNING: not subject-independent"
    cv_scores = cross_val_score(
        SVC(**SVM_PARAMS), X_pre, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='accuracy')
    overall_cv_acc = cv_scores.mean()
    ci_low, ci_high = cv_scores.min(), cv_scores.max()
    cv_report = "N/A"
    all_y_true, all_y_pred = y, y
    cm_path = ""
    print(f"[CV] Mean +/- Std: {{cv_scores.mean():.3f}} +/- {{cv_scores.std():.3f}}")

# --- 9. TRAIN FINAL MODEL ---
print("\n[TRAIN] Fitting final SVM on full dataset...")

scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

X_final, y_final = X_scaled, y
if HAS_SMOTE:
    min_count = min(np.bincount(y)[np.bincount(y) > 0])
    if min_count >= 6:
        smote_f = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
        X_final, y_final = smote_f.fit_resample(X_scaled, y)
        print(f"[TRAIN] SMOTE: {{len(y)}} -> {{len(y_final)}} samples")

model_final = SVC(**SVM_PARAMS)
model_final.fit(X_final, y_final)

train_acc = accuracy_score(y, model_final.predict(X_scaled))
gap = train_acc - cv_scores.mean()

print(f"[TRAIN] Train acc  : {{train_acc:.3f}}")
print(f"[TRAIN] CV acc     : {{cv_scores.mean():.3f}}")
print(f"[TRAIN] Gap        : {{gap:.3f}}  "
      f"{{'⚠️ overfitting' if gap > 0.10 else 'OK' if gap <= 0.05 else '⚠️ mild'}}")

# --- 10. SAVE ---
with open(MODEL_PATH,    'wb') as f: pickle.dump(model_final,  f)
with open(FEATURES_PATH, 'wb') as f: pickle.dump(feature_cols, f)
with open(SCALER_PATH,   'wb') as f: pickle.dump(scaler_final, f)

report = f"""MEDITATION CLASSIFIER — TRAINING REPORT (SVM v3)
=================================================
Dataset        : {{DATA_PATH}}
Samples        : {{len(y)}}
Subjects       : {{n_subjects}}
Features       : {{len(feature_cols}})
Channel strat  : {{channel_strategy}}

MODEL PARAMS (SVM RBF)
──────────────────────
  kernel          = rbf
  C               = {{best_C}}
  gamma           = {{best_gamma}}
  class_weight    = balanced
  probability     = True
  SMOTE           = {{'yes' if HAS_SMOTE else 'no (pip install imbalanced-learn)'}}

DIAGNOSTICS
───────────
  Depth-1 tree acc : {{dt1_acc:.3f}}  {{'⚠️ CIRCULAR' if label_circularity_warning else 'OK'}}
  Train acc        : {{train_acc:.3f}}
  CV acc           : {{cv_scores.mean():.3f}}
  Train-CV gap     : {{gap:.3f}}  {{'⚠️ overfitting' if gap > 0.10 else 'OK' if gap <= 0.05 else '⚠️ mild'}}

CV RESULTS ({{cv_label}})
────────────────────────
  Fold scores : {{np.round(cv_scores, 3).tolist()}}
  Mean +/- Std: {{cv_scores.mean():.3f}} +/- {{cv_scores.std():.3f}}
  95% CI      : [{{ci_low:.3f}}, {{ci_high:.3f}}]
  Overall     : {{overall_cv_acc:.3f}}

Labels: {{dict(zip(*np.unique(y, return_counts=True)))}}

{{cv_report}}
"""
with open(REPORT_PATH, 'w') as f:
    f.write(report)

print(f"\n[SAVE] Model   -> {{MODEL_PATH}}")
print(f"[SAVE] Feats   -> {{FEATURES_PATH}}")
print(f"[SAVE] Scaler  -> {{SCALER_PATH}}")
print(f"[SAVE] Report  -> {{REPORT_PATH}}")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE (SVM)")
print(f"  CV: {{cv_scores.mean():.3f}} +/- {{cv_scores.std():.3f}}  CI: [{{ci_low:.3f}}, {{ci_high:.3f}}]")
print(f"  Gap: {{gap:.3f}}")
if label_circularity_warning:
    print("  ⚠️  Labels likely circular — review labelling methodology")
print("=" * 70)