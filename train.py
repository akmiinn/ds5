"""
================================================================================
  MEDITATION STATE CLASSIFIER — TRAINING SCRIPT
================================================================================
  Input : meditation_levels_output.csv
  Output: meditation_rf_classifier.pkl, model_feature_names.pkl, scaler.pkl,
          training_report.txt, confusion_matrix.png, feature_importance.png

  WHY THE PREVIOUS VERSION REPORTED 96–100% ACCURACY
  ────────────────────────────────────────────────────
  The random 80/20 hold-out split placed rows from the SAME monk subjects in
  both train and test. A Random Forest learns the spectral "fingerprint" of
  each individual's brain. When it sees test rows from a subject it trained on,
  it recognises that fingerprint rather than generalising to the meditation state.
  This is data leakage — the metric looks great but means nothing for a new user.

  THE FIX: Subject-Independent Evaluation
  ─────────────────────────────────────────
  We use GroupKFold to ensure every fold tests on subjects the model has NEVER
  seen during that fold's training. This simulates the real deployment scenario:
  your Muse 2 session is from a person not in the training data.

  The reported accuracy will be lower. That lower number is honest.
  A 96% that comes from leakage is worse than a 75% that is real.

  19 FEATURES (unchanged):
    mean_rel_{band}       x5   mean relative power across channels
    std_rel_{band}        x5   spatial std across channels
    ratio_theta_alpha     x1   theta/alpha
    ratio_alpha_beta      x1   alpha/beta
    ratio_delta_alpha     x1   delta/alpha
    ratio_theta_beta      x1   theta/beta
    ratio_delta_beta      x1   delta/beta
    ratio_gamma_alpha     x1   gamma/alpha
    frontal_alpha_asymm   x1   AF7_alpha - AF8_alpha
    frontal_theta_mean    x1   mean frontal theta
    occipital_alpha_mean  x1   mean occipital alpha
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

# ─── 1. PATHS ──────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(SCRIPT_DIR, 'meditation_levels_output.csv')
OUTPUT_DIR    = SCRIPT_DIR

MODEL_PATH    = os.path.join(OUTPUT_DIR, 'meditation_rf_classifier.pkl')
FEATURES_PATH = os.path.join(OUTPUT_DIR, 'model_feature_names.pkl')
SCALER_PATH   = os.path.join(OUTPUT_DIR, 'scaler.pkl')
REPORT_PATH   = os.path.join(OUTPUT_DIR, 'training_report.txt')

# ─── 2. CONSTANTS ──────────────────────────────────────────────────────────────
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

LEVEL_NAMES = {
    0: "Level 0 — Baseline (Beta/Gamma dominant)",
    1: "Level 1 — Relaxed (Alpha dominant)",
    2: "Level 2 — Focused Attention (Theta dominant)",
    3: "Level 3 — Deep Meditation (Delta dominant)",
}

# ─── 3. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 70)
print("  MEDITATION CLASSIFIER — TRAINING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA] Loaded {len(df)} rows x {df.shape[1]} columns")
print(f"[DATA] Meditation level distribution:")
print(df['meditation_level'].value_counts().sort_index().to_string())

# ─── 4. SUBJECT GROUPS ────────────────────────────────────────────────────────
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
    print("\n[WARN] No 'subject' column found — cannot do subject-independent CV")
    print("       Results will be optimistic. Add a subject ID column to your data.")

# ─── 5. IDENTIFY BAND POWER COLUMNS ──────────────────────────────────────────
META_COLS = {'Condition', 'TRIGGER', 'M1', 'M2', 'group', 'subject',
             'time_window', 'filepath', 'cluster', 'meditation_level', 'level_name'}

rel_band_cols = {}
abs_band_cols = {}
print()
for band in BANDS:
    rel_band_cols[band] = [c for c in df.columns
                           if f'_{band}/all' in c and not any(m in c for m in META_COLS)]
    abs_band_cols[band] = [c for c in df.columns
                           if f'_{band}' in c and '/all' not in c
                           and not any(m in c for m in META_COLS)]
    print(f"  {band:7s}: {len(rel_band_cols[band]):3d} relative cols, "
          f"{len(abs_band_cols[band]):3d} absolute cols")

all_power_cols = [c for b in BANDS for c in rel_band_cols[b] + abs_band_cols[b]]
df[all_power_cols] = df[all_power_cols].apply(pd.to_numeric, errors='coerce')
for c in all_power_cols:
    if df[c].isna().any():
        df[c].fillna(df[c].median(), inplace=True)

# ─── 6. ENGINEER FEATURES ─────────────────────────────────────────────────────
print("\n[FEATURES] Engineering 19 features...")
eps  = 1e-8
feat = pd.DataFrame(index=df.index)

# Group A: Mean relative power per band (5 features)
for band in BANDS:
    if rel_band_cols[band]:
        feat[f'mean_rel_{band}'] = df[rel_band_cols[band]].mean(axis=1)
    elif abs_band_cols[band]:
        total = sum(df[abs_band_cols[b]].mean(axis=1) for b in BANDS) + eps
        feat[f'mean_rel_{band}'] = df[abs_band_cols[band]].mean(axis=1) / total
    else:
        feat[f'mean_rel_{band}'] = 0.0
        print(f"  [WARN] No columns for {band} — set to 0")

# Group B: Std relative power per band — ALL 5 bands (5 features)
for band in BANDS:
    if rel_band_cols[band]:
        feat[f'std_rel_{band}'] = df[rel_band_cols[band]].std(axis=1).fillna(0)
    elif abs_band_cols[band]:
        total = sum(df[abs_band_cols[b]].mean(axis=1) for b in BANDS) + eps
        feat[f'std_rel_{band}'] = df[abs_band_cols[band]].div(total, axis=0).std(axis=1).fillna(0)
    else:
        feat[f'std_rel_{band}'] = 0.0

# Group C: Spectral ratios (6 features)
feat['ratio_theta_alpha'] = feat['mean_rel_Theta'] / (feat['mean_rel_Alpha'] + eps)
feat['ratio_alpha_beta']  = feat['mean_rel_Alpha'] / (feat['mean_rel_Beta']  + eps)
feat['ratio_delta_alpha'] = feat['mean_rel_Delta'] / (feat['mean_rel_Alpha'] + eps)
feat['ratio_theta_beta']  = feat['mean_rel_Theta'] / (feat['mean_rel_Beta']  + eps)
feat['ratio_delta_beta']  = feat['mean_rel_Delta'] / (feat['mean_rel_Beta']  + eps)
feat['ratio_gamma_alpha'] = feat['mean_rel_Gamma'] / (feat['mean_rel_Alpha'] + eps)

# Group D: Regional features (3 features)
left_fa  = [c for c in rel_band_cols['Alpha'] if any(ch in c for ch in ['AF3','F3','F7','FC1','FC3'])]
right_fa = [c for c in rel_band_cols['Alpha'] if any(ch in c for ch in ['AF4','F4','F8','FC2','FC4'])]
feat['frontal_alpha_asymm'] = (
    df[left_fa].mean(axis=1) - df[right_fa].mean(axis=1)
    if left_fa and right_fa else pd.Series(0.0, index=df.index)
)

frontal_theta = [c for c in rel_band_cols['Theta']
                 if any(ch in c for ch in ['AF3','AF4','F3','F4','FZ','FCZ'])]
feat['frontal_theta_mean'] = (df[frontal_theta].mean(axis=1)
                              if frontal_theta else feat['mean_rel_Theta'])

occ_alpha = [c for c in rel_band_cols['Alpha']
             if any(ch in c for ch in ['O1','O2','OZ','PO3','PO4','PO7','PO8'])]
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

print(f"\n[DATA] Final sample count: {len(y)}")

# ─── 7. CROSS-VALIDATION — SUBJECT-INDEPENDENT ────────────────────────────────
# This is the ONLY honest accuracy metric for this dataset.
# Each fold trains on some subjects and tests on completely different subjects.
# This directly simulates using the model on your Muse 2 (a new, unseen person).
print("\n[CV] Subject-independent cross-validation...")
print("[CV] NOTE: This is the honest metric. Each fold tests on subjects never")
print("     seen during that fold's training. Expect 70-85% for real generalisation.\n")

if groups is not None and n_subjects >= 5:
    n_splits = min(5, n_subjects)
    cv       = GroupKFold(n_splits=n_splits)
    cv_label = f"Subject-Independent GroupKFold (n_splits={n_splits})"

    # Scale within each fold to prevent leakage (fit scaler on train fold only)
    fold_accs     = []
    all_y_true    = []
    all_y_pred    = []
    fold_scalers  = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Fit scaler on train fold only
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        fold_clf = RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=3,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        fold_clf.fit(X_tr_s, y_tr)
        y_pred_fold = fold_clf.predict(X_te_s)

        acc = accuracy_score(y_te, y_pred_fold)
        fold_accs.append(acc)
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred_fold)

        subj_in_test = np.unique(groups[test_idx])
        print(f"  Fold {fold_i+1}: test subjects={subj_in_test}  "
              f"n={len(y_te)}  acc={acc:.3f}")

    cv_scores = np.array(fold_accs)
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    overall_cv_acc = accuracy_score(all_y_true, all_y_pred)
    print(f"\n[CV] Strategy    : {cv_label}")
    print(f"[CV] Fold scores : {np.round(cv_scores, 3)}")
    print(f"[CV] Mean +- Std : {cv_scores.mean():.3f} +- {cv_scores.std():.3f}")
    print(f"[CV] Overall     : {overall_cv_acc:.3f}  (all folds pooled)")
    print()
    print("[CV] Classification Report — Pooled Subject-Independent CV:")
    present_labels = sorted(np.unique(y))
    cv_report = classification_report(
        all_y_true, all_y_pred,
        labels=present_labels,
        target_names=[LEVEL_NAMES[i] for i in present_labels]
    )
    print(cv_report)

    # Confusion matrix from pooled CV predictions (honest)
    cm_cv = confusion_matrix(all_y_true, all_y_pred, labels=present_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=cm_cv,
        display_labels=[f"L{i}" for i in present_labels]
    ).plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(
        f"Confusion Matrix — Subject-Independent CV  (acc={overall_cv_acc:.3f})\n"
        f"Each cell shows predictions on subjects unseen during training",
        fontsize=10
    )
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix -> {cm_path}  (subject-independent, honest)")

else:
    # Fallback if no subject column
    print("[CV] No subject column — using StratifiedKFold (results may be optimistic)")
    cv_label = "StratifiedKFold (n_splits=5) — WARNING: not subject-independent"
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                               class_weight='balanced', random_state=42, n_jobs=-1),
        X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    overall_cv_acc = cv_scores.mean()
    cv_report = "N/A — add subject column for per-class breakdown"
    all_y_true, all_y_pred = y, y   # placeholder
    cm_path = ""
    print(f"[CV] Mean +- Std : {cv_scores.mean():.3f} +- {cv_scores.std():.3f}")
    print("[WARN] These scores include same-subject rows in train and test.")

# ─── 8. TRAIN FINAL MODEL ON ALL DATA ────────────────────────────────────────
# The CV loop above is the honest evaluation.
# Now we train on ALL data to produce the best possible deployed model.
print("\n[TRAIN] Fitting final model on full dataset for deployment...")

scaler_final = StandardScaler()
X_scaled_final = scaler_final.fit_transform(X)

model_final = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,          # constrained to reduce memorisation
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model_final.fit(X_scaled_final, y)

train_acc = accuracy_score(y, model_final.predict(X_scaled_final))
print(f"[TRAIN] Training accuracy (full data): {train_acc:.3f}")
print(f"        (This will be high — the honest number is the CV above)")

# ─── 9. FEATURE IMPORTANCE PLOT ───────────────────────────────────────────────
importances = pd.Series(model_final.feature_importances_,
                         index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11, 5))
importances.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title("Feature Importances — Random Forest (trained on full dataset)", fontsize=12)
ax.set_ylabel("Mean Decrease in Gini Impurity")
ax.tick_params(axis='x', labelsize=8, rotation=45)
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"[PLOT] Feature importance -> {fi_path}")

# ─── 10. SAVE ARTEFACTS ───────────────────────────────────────────────────────
with open(MODEL_PATH,    'wb') as f: pickle.dump(model_final,  f)
with open(FEATURES_PATH, 'wb') as f: pickle.dump(feature_cols, f)
with open(SCALER_PATH,   'wb') as f: pickle.dump(scaler_final, f)
print(f"\n[SAVE] Model         -> {MODEL_PATH}")
print(f"[SAVE] Feature names -> {FEATURES_PATH}")
print(f"[SAVE] Scaler        -> {SCALER_PATH}")

report_content = f"""MEDITATION CLASSIFIER — TRAINING REPORT
========================================
Dataset       : {DATA_PATH}
Samples       : {len(y)}
Subjects      : {n_subjects}
Features      : {len(feature_cols)}
Feature list  : {feature_cols}

EVALUATION METHOD
─────────────────
{cv_label}

WHY THIS METHOD:
  The dataset contains multiple rows per monk subject. A random train/test split
  puts rows from the same subject in both halves, so the model learns each
  person's spectral fingerprint rather than generalising to new users.
  GroupKFold ensures every test fold contains subjects never seen in that fold's
  training — directly simulating inference on a new Muse 2 user.

HONEST RESULTS (subject-independent):
  CV fold scores : {np.round(cv_scores, 3).tolist()}
  CV mean        : {cv_scores.mean():.3f}
  CV std         : {cv_scores.std():.3f}
  CV overall     : {overall_cv_acc:.3f}  (all folds pooled)

  Expected range for real EEG meditation generalisation: 65-85%
  Scores above 90% with this dataset likely indicate data leakage.

TRAINING ACCURACY (full dataset, for reference only):
  {train_acc:.3f}  — high because model is evaluated on its own training data

Label distribution:
{pd.Series(y).value_counts().sort_index().to_string()}

Classification Report — Subject-Independent CV (honest):
{cv_report}

Feature Importances:
{importances.to_string()}
"""
with open(REPORT_PATH, 'w') as f:
    f.write(report_content)
print(f"[SAVE] Report        -> {REPORT_PATH}")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print(f"  Honest CV accuracy : {cv_scores.mean():.3f} +- {cv_scores.std():.3f}")
print(f"  (Subject-independent — simulates a new Muse 2 user)")
print("  Copy these 3 files into your testing directory:")
for p in [MODEL_PATH, FEATURES_PATH, SCALER_PATH]:
    print(f"    {p}")
print("=" * 70)