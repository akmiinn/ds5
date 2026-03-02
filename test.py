"""
================================================================================
  MEDITATION STATE CLASSIFIER — TESTING / INFERENCE SCRIPT (FIXED v2)
================================================================================
  Input : Muse 2 CSV with pre-computed band powers
          (Delta_TP9, Theta_AF7, Alpha_AF8, Beta_TP10, Gamma_*, ...)
           + RAW_TP9/AF7/AF8/TP10 columns (raw µV samples)
           + Accelerometer_X/Y/Z  (for motion artefact gating)

  Output: <filename>_predictions.csv
           <filename>_timeline.png

  FIXES
  ─────
  1. Bels→linear conversion: 10^value before computing relative power
  2. All 5 std_rel_ bands computed (was missing Gamma)
  3. Regional features use linear-converted values
  4. Relative paths, CLI argument support
  5. Confidence warnings for uncertain predictions
================================================================================
"""

import os
import sys
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── 1. PATHS ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
testing_dir = SCRIPT_DIR

# Accept target CSV from command line, or auto-detect S*.csv files
if len(sys.argv) > 1:
    target_file = sys.argv[1]
else:
    # Try to find any S*.csv file in the script directory
    candidates = sorted(glob.glob(os.path.join(SCRIPT_DIR, 'S47.csv')))
    if candidates:
        target_file = candidates[0]
        print(f"[INFO] Auto-detected: {target_file}")
        print(f"[INFO] Usage: python test.py <path_to_muse_csv>")
    else:
        print("[ERROR] No CSV specified and no S*.csv found in script directory.")
        print("        Usage: python test.py <path_to_muse_csv>")
        sys.exit(1)

model_path    = os.path.join(testing_dir, 'meditation_rf_classifier.pkl')
features_path = os.path.join(testing_dir, 'model_feature_names.pkl')
scaler_path   = os.path.join(testing_dir, 'scaler.pkl')

# ─── 2. CONSTANTS ──────────────────────────────────────────────────────────────
FS          = 256
WINDOW_SEC  = 2
STRIDE_SEC  = 1

MUSE_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
BANDS         = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

MOTION_THRESHOLD = 0.08

LEVEL_NAMES  = {0: "Baseline", 1: "Relaxed", 2: "Focused", 3: "Deep"}
LEVEL_COLORS = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71', 3: '#3498db'}

eps = 1e-8

# ─── 3. LOAD MODEL ARTEFACTS ───────────────────────────────────────────────────
print("=" * 70)
print("  MEDITATION CLASSIFIER — INFERENCE (FIXED v2)")
print("=" * 70)

for path, label in [(model_path, 'Model'),
                    (features_path, 'Feature names'),
                    (scaler_path,   'Scaler')]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] {label} not found: {path}\n"
            "        Run train.py first to generate .pkl files."
        )

with open(model_path,    'rb') as f: model         = pickle.load(f)
with open(features_path, 'rb') as f: feature_names = pickle.load(f)
with open(scaler_path,   'rb') as f: scaler        = pickle.load(f)

print(f"[LOAD] Model   : {model_path}")
print(f"[LOAD] Features: {len(feature_names)} features")
print(f"[LOAD] Scaler  : {scaler_path}")

# ─── 4. LOAD MUSE 2 CSV ────────────────────────────────────────────────────────
print(f"\n[DATA] Loading: {target_file}")
if not os.path.exists(target_file):
    raise FileNotFoundError(f"[ERROR] CSV not found: {target_file}")

raw_df = pd.read_csv(target_file, low_memory=False)
raw_df.dropna(how='all', inplace=True)
raw_df.reset_index(drop=True, inplace=True)
print(f"[DATA] Shape: {raw_df.shape}")

# ─── 4a. Detect band-power columns ────────────────────────────────────────────
band_cols = {}
for band in BANDS:
    band_cols[band] = {}
    for ch in MUSE_CHANNELS:
        col = f'{band}_{ch}'
        if col in raw_df.columns:
            band_cols[band][ch] = col
        else:
            alts = [c for c in raw_df.columns
                    if band.lower() in c.lower() and ch.lower() in c.lower()]
            if alts:
                band_cols[band][ch] = alts[0]

found_bands = {b: list(band_cols[b].values()) for b in BANDS if band_cols[b]}
for b, cols in found_bands.items():
    print(f"  {b:7s}: {cols}")

if not found_bands:
    raise ValueError(
        f"[ERROR] No band-power columns found.\n        Columns: {list(raw_df.columns)}"
    )

all_bp_cols = [c for b in BANDS for c in band_cols[b].values()]
raw_df[all_bp_cols] = raw_df[all_bp_cols].apply(pd.to_numeric, errors='coerce')

# ─── 4b. RAW EEG columns ──────────────────────────────────────────────────────
raw_eeg_cols = {}
for ch in MUSE_CHANNELS:
    col = f'RAW_{ch}'
    if col in raw_df.columns:
        raw_eeg_cols[ch] = col

# ─── 4c. IMU columns ──────────────────────────────────────────────────────────
imu_cols = [c for c in ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
            if c in raw_df.columns]
if not imu_cols:
    imu_cols = [c for c in raw_df.columns
                if 'acc' in c.lower() or 'accel' in c.lower()]
has_imu = len(imu_cols) >= 1
if has_imu:
    raw_df[imu_cols] = raw_df[imu_cols].apply(pd.to_numeric, errors='coerce')
    print(f"[DATA] IMU: {imu_cols}")
else:
    print("[DATA] No IMU — motion gating disabled")

# ─── 4d. Timestamps & sample rate ─────────────────────────────────────────────
time_col = None
for c in ['TimeStamp', 'timestamps', 'Timestamp', 'time', 'Time']:
    if c in raw_df.columns:
        time_col = c
        break

if time_col:
    raw_df[time_col] = pd.to_datetime(raw_df[time_col], errors='coerce')
    raw_df.dropna(subset=[time_col], inplace=True)
    raw_df.reset_index(drop=True, inplace=True)

    if len(raw_df) > 100:
        ts_sec = raw_df[time_col].astype(np.int64) / 1e9
        dt = np.diff(ts_sec[:500])
        dt = dt[(dt > 0) & (dt < 1.0)]
        if len(dt) > 10:
            est_fs = int(round(1.0 / np.median(dt)))
            if 50 <= est_fs <= 512:
                FS = est_fs
                print(f"[DATA] Sample rate: {FS} Hz")

WINDOW_SAMP = FS * WINDOW_SEC
STRIDE_SAMP = FS * STRIDE_SEC
print(f"[DATA] {len(raw_df)} rows  (~{len(raw_df)/FS:.0f}s at {FS} Hz)")

# ─── 4e. Quality flags ────────────────────────────────────────────────────────
hsi_cols = [c for c in raw_df.columns if c.startswith('HSI_')]
headband_col = 'HeadBandOn' if 'HeadBandOn' in raw_df.columns else None

if headband_col:
    raw_df[headband_col] = pd.to_numeric(raw_df[headband_col], errors='coerce')

quality_ok = pd.Series(True, index=raw_df.index)
if headband_col:
    quality_ok &= (raw_df[headband_col] == 1)
    n_off = (~quality_ok).sum()
    if n_off:
        print(f"[QUALITY] {n_off} rows HeadBandOn=0")

if hsi_cols:
    raw_df[hsi_cols] = raw_df[hsi_cols].apply(pd.to_numeric, errors='coerce')
    hsi_bad = (raw_df[hsi_cols] > 2).any(axis=1)
    quality_ok &= ~hsi_bad
    print(f"[QUALITY] {hsi_bad.sum()} rows poor HSI")

# ─── 5. WINDOWING & FEATURE EXTRACTION ────────────────────────────────────────
print(f"\n[FEATURES] Windowing ({WINDOW_SEC}s window, {STRIDE_SEC}s stride)...")

n_rows = len(raw_df)
n_win  = max(0, (n_rows - WINDOW_SAMP) // STRIDE_SAMP + 1)
print(f"[FEATURES] Expected windows: {n_win}")

records       = []
window_times  = []
skipped_motion  = 0
skipped_quality = 0


def get_ch_rel_power(win_df, band_name, channel_name, total_pwr):
    """Convert one channel's band power from Bels→linear→relative."""
    if channel_name in band_cols[band_name]:
        vals = pd.to_numeric(win_df[band_cols[band_name][channel_name]],
                             errors='coerce').dropna()
        if len(vals) > 0:
            return float((10.0 ** vals).mean()) / total_pwr
    return np.nan


for i in range(n_win):
    start = i * STRIDE_SAMP
    end   = start + WINDOW_SAMP
    win   = raw_df.iloc[start:end]

    # Quality gate
    if quality_ok.iloc[start:end].mean() < 0.70:
        skipped_quality += 1
        continue

    # Motion gate
    if has_imu:
        imu_vals = win[imu_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        resultant = np.sqrt((imu_vals ** 2).sum(axis=1))
        if (resultant - resultant.mean()).std() > MOTION_THRESHOLD:
            skipped_motion += 1
            continue

    # ── Convert band powers: Bels → linear → relative ──
    row = {}
    bp_linear = {}
    for band in BANDS:
        ch_vals = []
        for ch, col in band_cols[band].items():
            vals = pd.to_numeric(win[col], errors='coerce').dropna()
            if len(vals) > 0:
                ch_vals.append(float((10.0 ** vals).mean()))
        bp_linear[band] = np.array(ch_vals) if ch_vals else np.array([0.0])

    total_power = sum(bp_linear[b].mean() for b in BANDS) + eps

    # Group A: mean relative power (5)
    for band in BANDS:
        row[f'mean_rel_{band}'] = float(bp_linear[band].mean()) / total_power

    # Group B: std relative power — ALL 5 bands (5)
    for band in BANDS:
        if len(bp_linear[band]) > 1:
            row[f'std_rel_{band}'] = float((bp_linear[band] / total_power).std())
        else:
            row[f'std_rel_{band}'] = 0.0

    # Group C: spectral ratios (6)
    row['ratio_theta_alpha'] = row['mean_rel_Theta'] / (row['mean_rel_Alpha'] + eps)
    row['ratio_alpha_beta']  = row['mean_rel_Alpha'] / (row['mean_rel_Beta']  + eps)
    row['ratio_delta_alpha'] = row['mean_rel_Delta'] / (row['mean_rel_Alpha'] + eps)
    row['ratio_theta_beta']  = row['mean_rel_Theta'] / (row['mean_rel_Beta']  + eps)
    row['ratio_delta_beta']  = row['mean_rel_Delta'] / (row['mean_rel_Beta']  + eps)
    row['ratio_gamma_alpha'] = row['mean_rel_Gamma'] / (row['mean_rel_Alpha'] + eps)

    # Group D: regional features (3)
    af7_a = get_ch_rel_power(win, 'Alpha', 'AF7', total_power)
    af8_a = get_ch_rel_power(win, 'Alpha', 'AF8', total_power)
    row['frontal_alpha_asymm'] = (
        (af7_a - af8_a) if not np.isnan(af7_a) and not np.isnan(af8_a) else 0.0
    )

    ft = [get_ch_rel_power(win, 'Theta', ch, total_power) for ch in ['AF7', 'AF8']]
    ft = [v for v in ft if not np.isnan(v)]
    row['frontal_theta_mean'] = float(np.mean(ft)) if ft else row['mean_rel_Theta']

    oa = [get_ch_rel_power(win, 'Alpha', ch, total_power) for ch in ['TP9', 'TP10']]
    oa = [v for v in oa if not np.isnan(v)]
    row['occipital_alpha_mean'] = float(np.mean(oa)) if oa else row['mean_rel_Alpha']

    if time_col:
        window_times.append(raw_df[time_col].iloc[start])
    else:
        window_times.append(start / FS)

    records.append(row)

print(f"[FEATURES] Valid    : {len(records)}")
print(f"[FEATURES] Skipped  : {skipped_motion} motion, {skipped_quality} quality")

if not records:
    raise RuntimeError("[ERROR] No valid windows. Check signal quality.")

feat_df = pd.DataFrame(records)

# ─── 6. ALIGN FEATURES ───────────────────────────────────────────────────────
missing = [f for f in feature_names if f not in feat_df.columns]
if missing:
    print(f"[WARN] {len(missing)} missing features (filled 0): {missing}")
    for m in missing:
        feat_df[m] = 0.0

X_test = feat_df[feature_names].values.astype(np.float64)
bad = ~np.isfinite(X_test)
if bad.any():
    print(f"[WARN] {bad.sum()} NaN/Inf → 0")
    X_test[bad] = 0.0

# ─── 7. PREDICT ──────────────────────────────────────────────────────────────
X_scaled     = scaler.transform(X_test)
predictions  = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
confidence   = probabilities.max(axis=1)

print("\n[PREDICT] Distribution:")
for lvl in sorted(LEVEL_NAMES):
    mask = predictions == lvl
    n = mask.sum()
    pct = 100 * n / len(predictions)
    bar = '█' * int(pct / 2)
    print(f"  L{lvl} {LEVEL_NAMES[lvl]:10s}: {n:4d} ({pct:5.1f}%)  {bar}")

# ─── 8. RESULTS ─────────────────────────────────────────────────────────────
results = feat_df[feature_names].copy()
results.insert(0, 'window_time', window_times[:len(results)])
results['predicted_level'] = predictions
results['level_name']      = [LEVEL_NAMES.get(p, str(p)) for p in predictions]
results['confidence']      = np.round(confidence, 4)
n_classes = probabilities.shape[1]
for i, lvl in enumerate(sorted(LEVEL_NAMES)[:n_classes]):
    results[f'prob_L{lvl}'] = np.round(probabilities[:, i], 4)

# ─── 9. SUMMARY ─────────────────────────────────────────────────────────────
total_s = len(raw_df) / FS
print(f"\n[SUMMARY] Duration    : {total_s:.0f}s ({total_s/60:.1f} min)")
print(f"[SUMMARY] Confidence  : {confidence.mean():.3f} mean, "
      f"{np.median(confidence):.3f} median")
dominant = LEVEL_NAMES.get(int(np.bincount(predictions).argmax()), '?')
print(f"[SUMMARY] Dominant    : {dominant}")
for lvl in sorted(LEVEL_NAMES):
    secs = (predictions == lvl).sum() * STRIDE_SEC
    print(f"  L{lvl} {LEVEL_NAMES[lvl]:10s}: {secs:5.0f}s  ({100*secs/max(1,total_s):.1f}%)")

# Confidence analysis
low_conf  = (confidence < 0.50).sum()
med_conf  = ((confidence >= 0.50) & (confidence < 0.70)).sum()
high_conf = (confidence >= 0.70).sum()
print(f"\n[CONFIDENCE] High (≥70%): {high_conf} ({100*high_conf/len(confidence):.1f}%)")
print(f"[CONFIDENCE] Med (50-70%): {med_conf} ({100*med_conf/len(confidence):.1f}%)")
print(f"[CONFIDENCE] Low (<50%)  : {low_conf} ({100*low_conf/len(confidence):.1f}%)")

if low_conf / len(confidence) > 0.20:
    print("\n⚠️  >20% low-confidence windows. Possible causes:")
    print("   • Poor electrode contact during recording")
    print("   • Domain gap: model trained on 64-ch research EEG, you use 4-ch Muse")
    print("   • Subject's EEG pattern differs from training population (monks)")
    print("   Suggestion: focus analysis on HIGH-confidence windows only.")
elif low_conf > 0:
    print(f"\n[INFO] {low_conf} low-confidence windows — normal for consumer EEG.")

# ─── 10. SAVE ────────────────────────────────────────────────────────────────
base = os.path.splitext(os.path.basename(target_file))[0]
out_csv = os.path.join(testing_dir, f'{base}_predictions.csv')
results.to_csv(out_csv, index=False)
print(f"\n[SAVE] CSV  → {out_csv}")

# ─── 11. PLOT ────────────────────────────────────────────────────────────────
if pd.api.types.is_datetime64_any_dtype(pd.Series(window_times)):
    t0 = pd.Timestamp(window_times[0])
    times = np.array([(pd.Timestamp(t) - t0).total_seconds()
                      for t in window_times[:len(predictions)]])
else:
    times = np.array(window_times[:len(predictions)], dtype=float)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Subplot 1: Level
ax = axes[0]
for lvl in sorted(LEVEL_NAMES):
    mask = predictions == lvl
    if mask.any():
        ax.scatter(times[mask], np.full(mask.sum(), lvl),
                   color=LEVEL_COLORS[lvl], s=12, alpha=0.8,
                   label=f"L{lvl}: {LEVEL_NAMES[lvl]}")
ax.set_ylabel("Level")
ax.set_yticks(list(LEVEL_NAMES.keys()))
ax.set_yticklabels([f"L{k}" for k in LEVEL_NAMES.keys()])
ax.set_title("Predicted Meditation Level")
ax.legend(ncol=4, fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# Subplot 2: Confidence (color-coded)
ax = axes[1]
ax.fill_between(times, confidence, alpha=0.15, color='purple')
ax.plot(times, confidence, color='purple', linewidth=0.9, alpha=0.85)
ax.axhline(0.70, linestyle='--', color='green', linewidth=0.8, label='70% (high)')
ax.axhline(0.50, linestyle='--', color='red', linewidth=0.8, label='50% (low)')
ax.set_ylabel("Confidence")
ax.set_ylim(0, 1.05)
ax.set_title("Model Confidence (windows below red line are unreliable)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Subplot 3: Band powers
ax = axes[2]
band_colors = {'Delta': '#e74c3c', 'Theta': '#f39c12',
               'Alpha': '#2ecc71', 'Beta':  '#3498db', 'Gamma': '#9b59b6'}
for band in BANDS:
    col = f'mean_rel_{band}'
    if col in results.columns:
        ax.plot(times, results[col].values,
                label=band, color=band_colors[band], linewidth=0.9, alpha=0.85)
ax.set_ylabel("Relative Power")
ax.set_xlabel("Time (seconds)")
ax.set_title("Band Power Over Time")
ax.legend(fontsize=8, ncol=5)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(testing_dir, f'{base}_timeline.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[SAVE] Plot → {plot_path}")

print("\n" + "=" * 70)
print("  INFERENCE COMPLETE")
print(f"  CSV  : {out_csv}")
print(f"  Plot : {plot_path}")
print("=" * 70)    ratio_gamma_alpha     x1   gamma/alpha
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     LeaveOneGroupOut)
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

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

# ─── Muse 2 channel mapping ────────────────────────────────────────────────────
# Muse 2 has 4 channels: TP9, AF7, AF8, TP10
# Map each to the closest 10-20 system channels found in research EEG data.
# This reduces the domain gap between 64-ch training and 4-ch inference.
MUSE_CHANNEL_MAP = {
    'TP9':  ['TP7', 'T7', 'TP9', 'P7', 'P9'],        # left temporal-parietal
    'AF7':  ['AF3', 'F7', 'AF7', 'Fp1', 'F5', 'F3'],  # left frontal
    'AF8':  ['AF4', 'F8', 'AF8', 'Fp2', 'F6', 'F4'],  # right frontal
    'TP10': ['TP8', 'T8', 'TP10', 'P8', 'P10'],       # right temporal-parietal
}

# For frontal alpha asymmetry — left vs right frontal channels
FRONTAL_LEFT_CHANNELS  = ['AF3', 'F3', 'F7', 'AF7', 'Fp1', 'F5', 'FC1', 'FC3']
FRONTAL_RIGHT_CHANNELS = ['AF4', 'F4', 'F8', 'AF8', 'Fp2', 'F6', 'FC2', 'FC4']

# For frontal theta — frontal midline + laterals
FRONTAL_THETA_CHANNELS = ['AF3', 'AF4', 'AF7', 'AF8', 'F3', 'F4', 'FZ', 'FCZ',
                          'Fp1', 'Fp2', 'F7', 'F8']

# For occipital/posterior alpha — closest to TP9/TP10 on Muse
POSTERIOR_ALPHA_CHANNELS = ['O1', 'O2', 'OZ', 'PO3', 'PO4', 'PO7', 'PO8',
                            'P7', 'P8', 'TP7', 'TP8', 'TP9', 'TP10', 'P9', 'P10']


# ─── 3. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 70)
print("  MEDITATION CLASSIFIER — TRAINING (FIXED v2)")
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
    print("\n[WARN] No 'subject' column — cannot do subject-independent CV")

# ─── 5. IDENTIFY BAND POWER COLUMNS ──────────────────────────────────────────
META_COLS = {'Condition', 'TRIGGER', 'M1', 'M2', 'group', 'subject',
             'time_window', 'filepath', 'cluster', 'meditation_level', 'level_name'}

rel_band_cols = {}
abs_band_cols = {}

# Also build Muse-proximity columns: only channels near Muse 2 positions
muse_rel_band_cols = {}
muse_abs_band_cols = {}

# All Muse-mapped channel names (flattened)
all_muse_ch = set()
for ch_list in MUSE_CHANNEL_MAP.values():
    all_muse_ch.update([c.upper() for c in ch_list])

print()
for band in BANDS:
    # ALL relative and absolute columns for this band
    rel_band_cols[band] = [c for c in df.columns
                           if f'_{band}/all' in c
                           and not any(m in c for m in META_COLS)]
    abs_band_cols[band] = [c for c in df.columns
                           if f'_{band}' in c and '/all' not in c
                           and not any(m in c for m in META_COLS)]

    # MUSE-PROXIMITY: only keep columns whose channel name is in our mapping
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

# Decide which column set to use:
# If Muse-proximity columns exist for most bands, use them to reduce domain gap.
# Otherwise fall back to all columns.
muse_coverage = sum(1 for b in BANDS if muse_rel_band_cols[b] or muse_abs_band_cols[b])
if muse_coverage >= 4:
    use_rel = muse_rel_band_cols
    use_abs = muse_abs_band_cols
    channel_strategy = "Muse-proximity (reduced domain gap)"
    print(f"\n[STRATEGY] Using Muse-proximity channels ({muse_coverage}/5 bands covered)")
    print("[STRATEGY] This reduces the gap between 64-ch training and 4-ch Muse inference")
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

# ─── 6. ENGINEER FEATURES ─────────────────────────────────────────────────────
print("\n[FEATURES] Engineering 19 features...")
eps  = 1e-8
feat = pd.DataFrame(index=df.index)

# Group A: Mean relative power per band (5 features)
for band in BANDS:
    if use_rel[band]:
        feat[f'mean_rel_{band}'] = df[use_rel[band]].mean(axis=1)
    elif use_abs[band]:
        total = sum(df[use_abs[b]].mean(axis=1) for b in BANDS) + eps
        feat[f'mean_rel_{band}'] = df[use_abs[band]].mean(axis=1) / total
    else:
        feat[f'mean_rel_{band}'] = 0.0
        print(f"  [WARN] No columns for {band} — set to 0")

# Group B: Std relative power per band — ALL 5 bands (5 features)
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

# Group C: Spectral ratios (6 features)
feat['ratio_theta_alpha'] = feat['mean_rel_Theta'] / (feat['mean_rel_Alpha'] + eps)
feat['ratio_alpha_beta']  = feat['mean_rel_Alpha'] / (feat['mean_rel_Beta']  + eps)
feat['ratio_delta_alpha'] = feat['mean_rel_Delta'] / (feat['mean_rel_Alpha'] + eps)
feat['ratio_theta_beta']  = feat['mean_rel_Theta'] / (feat['mean_rel_Beta']  + eps)
feat['ratio_delta_beta']  = feat['mean_rel_Delta'] / (feat['mean_rel_Beta']  + eps)
feat['ratio_gamma_alpha'] = feat['mean_rel_Gamma'] / (feat['mean_rel_Alpha'] + eps)

# Group D: Regional features (3 features)
# Use Muse-specific frontal channels for asymmetry
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
      f"{ {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")

# ─── 6b. LABEL CIRCULARITY DIAGNOSTIC ─────────────────────────────────────────
print("\n[DIAGNOSTIC] Checking for label circularity...")

for d in [1, 2, 3]:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X, y)
    print(f"  Decision tree (depth={d}) train acc: {accuracy_score(y, dt.predict(X)):.3f}")

dt1 = DecisionTreeClassifier(max_depth=1, random_state=42)
dt1.fit(X, y)
dt1_acc = accuracy_score(y, dt1.predict(X))
label_circularity_warning = dt1_acc > 0.70

if label_circularity_warning:
    print(f"\n  ⚠️  Depth-1 tree gets {dt1_acc:.1%}. Labels are likely derived from")
    print("     the band-power features. Model learns a circular mapping.")
    print("     Results will look good on paper but won't generalise well to")
    print("     truly novel EEG data from different recording conditions.\n")
else:
    print(f"\n  ✓ No obvious circularity (depth-1 acc = {dt1_acc:.1%})\n")

# ─── 7. CROSS-VALIDATION ──────────────────────────────────────────────────────
print("─" * 70)
print("[CV] Subject-independent cross-validation\n")

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=8,
    min_samples_split=15,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
print(f"[CV] RF params: depth={RF_PARAMS['max_depth']}, "
      f"leaf≥{RF_PARAMS['min_samples_leaf']}, "
      f"split≥{RF_PARAMS['min_samples_split']}, "
      f"features={RF_PARAMS['max_features']}")

if groups is not None and n_subjects >= 5:
    n_splits = min(5, n_subjects)
    cv       = GroupKFold(n_splits=n_splits)
    cv_label = f"Subject-Independent GroupKFold (n_splits={n_splits})"

    fold_accs    = []
    all_y_true   = []
    all_y_pred   = []

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

        fold_clf = RandomForestClassifier(**RF_PARAMS)
        fold_clf.fit(X_tr_s, y_tr)
        y_pred_fold = fold_clf.predict(X_te_s)

        acc = accuracy_score(y_te, y_pred_fold)
        fold_accs.append(acc)
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred_fold)

        subj_test = np.unique(groups[test_idx])
        dist = dict(zip(*np.unique(y_te, return_counts=True)))
        print(f"  Fold {fold_i+1}: subjects={subj_test}  n={len(y_te)}  "
              f"acc={acc:.3f}  dist={dist}")

    cv_scores = np.array(fold_accs)
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

    print(f"\n[CV] Strategy    : {cv_label}")
    print(f"[CV] Folds       : {np.round(cv_scores, 3)}")
    print(f"[CV] Mean ± Std  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"[CV] 95% CI      : [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"[CV] Overall     : {overall_cv_acc:.3f}")
    if HAS_SMOTE:
        print(f"[CV] SMOTE       : applied per fold")

    present_labels = sorted(np.unique(y))
    cv_report = classification_report(
        all_y_true, all_y_pred,
        labels=present_labels,
        target_names=[LEVEL_NAMES[i] for i in present_labels]
    )
    print(f"\n{cv_report}")

    for lvl in present_labels:
        n_test = (all_y_true == lvl).sum()
        if n_test < 20:
            print(f"  ⚠️  Class {lvl}: only {n_test} test samples — metric unreliable")

    cm_cv = confusion_matrix(all_y_true, all_y_pred, labels=present_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=cm_cv,
        display_labels=[f"L{i}" for i in present_labels]
    ).plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(f"Confusion Matrix — Subject-Independent CV (acc={overall_cv_acc:.3f})",
                 fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()

else:
    print("[CV] No subject column — StratifiedKFold (optimistic)")
    cv_label = "StratifiedKFold — WARNING: not subject-independent"
    cv_scores = cross_val_score(
        RandomForestClassifier(**RF_PARAMS), X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='accuracy')
    overall_cv_acc = cv_scores.mean()
    ci_low, ci_high = cv_scores.min(), cv_scores.max()
    cv_report = "N/A"
    all_y_true, all_y_pred = y, y
    cm_path = ""
    print(f"[CV] Mean ± Std : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ─── 8. TRAIN FINAL MODEL ────────────────────────────────────────────────────
print("\n[TRAIN] Fitting final model on full dataset...")

scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

X_final, y_final = X_scaled, y
if HAS_SMOTE:
    min_count = min(np.bincount(y)[np.bincount(y) > 0])
    if min_count >= 6:
        smote_f = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
        X_final, y_final = smote_f.fit_resample(X_scaled, y)
        print(f"[TRAIN] SMOTE: {len(y)} → {len(y_final)} samples")

model_final = RandomForestClassifier(**RF_PARAMS)
model_final.fit(X_final, y_final)

train_acc = accuracy_score(y, model_final.predict(X_scaled))
gap = train_acc - cv_scores.mean()

print(f"[TRAIN] Train acc  : {train_acc:.3f}")
print(f"[TRAIN] CV acc     : {cv_scores.mean():.3f}")
print(f"[TRAIN] Gap        : {gap:.3f}  "
      f"{'⚠️ overfitting' if gap > 0.10 else '✓ OK' if gap <= 0.05 else '⚠️ mild'}")

# ─── 9. FEATURE IMPORTANCE ───────────────────────────────────────────────────
importances = pd.Series(model_final.feature_importances_,
                         index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11, 5))
importances.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title("Feature Importances — Constrained RF", fontsize=12)
ax.set_ylabel("Gini Importance")
ax.tick_params(axis='x', labelsize=8, rotation=45)
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
plt.savefig(fi_path, dpi=150)
plt.close()

# ─── 10. SAVE ─────────────────────────────────────────────────────────────────
with open(MODEL_PATH,    'wb') as f: pickle.dump(model_final,  f)
with open(FEATURES_PATH, 'wb') as f: pickle.dump(feature_cols, f)
with open(SCALER_PATH,   'wb') as f: pickle.dump(scaler_final, f)

report = f"""MEDITATION CLASSIFIER — TRAINING REPORT (FIXED v2)
===================================================
Dataset        : {DATA_PATH}
Samples        : {len(y)}
Subjects       : {n_subjects}
Features       : {len(feature_cols)}
Channel strat  : {channel_strategy}

MODEL PARAMS
────────────
  n_estimators    = {RF_PARAMS['n_estimators']}
  max_depth       = {RF_PARAMS['max_depth']}
  min_samples_leaf = {RF_PARAMS['min_samples_leaf']}
  min_samples_split = {RF_PARAMS['min_samples_split']}
  max_features    = {RF_PARAMS['max_features']}
  SMOTE           = {'yes' if HAS_SMOTE else 'no (pip install imbalanced-learn)'}

DIAGNOSTICS
───────────
  Depth-1 tree acc : {dt1_acc:.3f}  {'⚠️ CIRCULAR' if label_circularity_warning else '✓ OK'}
  Train acc        : {train_acc:.3f}
  CV acc           : {cv_scores.mean():.3f}
  Train–CV gap     : {gap:.3f}  {'⚠️ overfitting' if gap > 0.10 else '✓ OK' if gap <= 0.05 else '⚠️ mild'}

CV RESULTS ({cv_label})
────────────────────────
  Fold scores : {np.round(cv_scores, 3).tolist()}
  Mean ± Std  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}
  95% CI      : [{ci_low:.3f}, {ci_high:.3f}]
  Overall     : {overall_cv_acc:.3f}

Labels: {dict(zip(*np.unique(y, return_counts=True)))}

{cv_report}

Feature Importances:
{importances.to_string()}
"""
with open(REPORT_PATH, 'w') as f:
    f.write(report)

print(f"\n[SAVE] Model   → {MODEL_PATH}")
print(f"[SAVE] Feats   → {FEATURES_PATH}")
print(f"[SAVE] Scaler  → {SCALER_PATH}")
print(f"[SAVE] Report  → {REPORT_PATH}")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print(f"  CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  CI: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"  Gap: {gap:.3f}")
if label_circularity_warning:
    print("  ⚠️  Labels likely circular — review labelling methodology")
print("=" * 70)
