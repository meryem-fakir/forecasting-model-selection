import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from scipy.optimize import minimize

# ── Data ─────────────────────────────────────────────────────────────────────
df1 = pd.read_csv("selected_models.csv").reset_index(drop=True)
df2 = pd.read_csv("BWM_weights.csv").reset_index(drop=True)

merged_df = pd.merge(df1, df2, left_on="sample", right_on="series", how="inner")
merged = df1.merge(df2, left_on='sample', right_on='series', how='inner').reset_index(drop=True)

MODELS = ["SARIMA", "ETS", "BSTS", "RandomForest", "GradientBoosting"]
FEATURES = ["w_trend", "w_seasonality", "w_curvature", "w_spikiness"]

mean = merged_df.groupby("min_rmse")[FEATURES].mean().loc[MODELS].round(4)
var  = merged_df.groupby("min_rmse")[FEATURES].var().loc[MODELS].round(4)

result = mean.astype(str) + " / " + var.astype(str)
print(result.to_string())

w_star = merged[FEATURES].values.astype(float)
Y_str  = merged["min_rmse"].values
classes   = sorted(set(Y_str))
label2idx = {m: i for i, m in enumerate(classes)}
Y         = np.array([label2idx[y] for y in Y_str])
N         = len(Y)
J         = len(classes)
K         = len(FEATURES)

# ── Bounds from real data (mean ± sqrt(variance)) ────────────────────────────
stats = {m: [(mean.loc[m, f], var.loc[m, f]) for f in FEATURES] for m in MODELS}

lo = np.array([[m - np.sqrt(v) for m, v in row] for row in stats.values()])
hi = np.array([[m + np.sqrt(v) for m, v in row] for row in stats.values()])

class_to_row = {"SARIMA": 0, "ETS": 1, "BSTS": 2, "RandomForest": 3, "GradientBoosting": 4}
lo = np.array([lo[class_to_row[c]] for c in classes])
hi = np.array([hi[class_to_row[c]] for c in classes])

sigma_P = np.array([[np.sqrt(v) for m, v in stats[c]] for c in classes])

# ── Target counts ─────────────────────────────────────────────────────────────
real_counts_named = df1["min_rmse"].value_counts().reindex(MODELS).fillna(0).astype(int).to_dict()
target_counts = np.array([real_counts_named[c] for c in classes], dtype=float)

# ── Fitness ────────────────────────────────────────────────────────────────────
alpha = 0.7

def fitness(P):
    scores = w_star @ P.T
    pred   = np.argmax(scores, axis=1)
    class_error = np.mean(pred != Y)
    cnts       = np.bincount(pred, minlength=J).astype(float)
    dist_error = np.sum((cnts - target_counts) ** 2) / (N ** 2)
    return alpha * class_error + (1 - alpha) * dist_error

# ── Simulated Annealing ───────────────────────────────────────────────────────
T     = 1.0
T_min = 1e-10
rate  = 0.995

P_cur = np.array([[m for m, v in row] for row in stats.values()])
P_cur = np.array([P_cur[class_to_row[c]] for c in classes])
f_cur = fitness(P_cur)
P_best, f_best = P_cur.copy(), f_cur

it = 0
while T > T_min:
    r, k        = np.random.randint(0, J), np.random.randint(0, K)
    P_new       = P_cur.copy()
    P_new[r, k] = np.clip(P_cur[r, k] + np.random.normal(0, sigma_P[r, k]), lo[r, k], hi[r, k])

    f_new = fitness(P_new)
    delta = f_new - f_cur

    if delta < 0 or np.random.rand() < np.exp(-delta / T):
        P_cur, f_cur = P_new, f_new

    if f_cur < f_best:
        P_best, f_best = P_cur.copy(), f_cur

    T  *= rate
    it += 1

    if it % 5000 == 0:
        pred = np.argmax(w_star @ P_best.T, axis=1)
        cnts = np.bincount(pred, minlength=J)
        acc  = np.mean(pred == Y) * 100
        print(f"  iter={it:>7}  T={T:.5f}  fitness={f_best:.4f}  acc={acc:.1f}%  counts={cnts}")

print(f"\n✓ Done — best fitness: {f_best:.4f}")

# ── Final results ─────────────────────────────────────────────────────────────
scores    = w_star @ P_best.T
ranks     = np.argsort(-scores, axis=1)
pred_best = np.array([classes[i] for i in np.argmax(scores, axis=1)])

real_cnt = pd.Series(Y_str).value_counts().reindex(MODELS).fillna(0).astype(int)
pred_cnt = pd.Series(pred_best).value_counts().reindex(MODELS).fillna(0).astype(int)

print("\n  — REAL —")
for m in MODELS:
    r = real_cnt[m]
    print(f"  {m:<20} {r:>5}  {'█' * (r // 20)}")

print("\n  — PREDICTED —")
for m in MODELS:
    p = pred_cnt[m]
    print(f"  {m:<20} {p:>5}  {'█' * (p // 20)}")

print("\n" + "=" * 60)
print("  OPTIMISED P MATRIX")
print("=" * 60)
for i, c in enumerate(classes):
    print(f"  {c:<20} {np.round(P_best[i], 4)}")


print("\n" + "=" * 60)
print(" LO MATRIX (mean - sqrt(variance))")
print("=" * 60)
df_lo = pd.DataFrame(lo, index=classes, columns=FEATURES)
print(df_lo.round(4).to_string())

print("\n" + "=" * 60)
print(" HI MATRIX (mean + sqrt(variance))")
print("=" * 60)
df_hi = pd.DataFrame(hi, index=classes, columns=FEATURES)
print(df_hi.round(4).to_string())