import pandas as pd
import matplotlib.pyplot as plt

# Load data and handle missing values
df = pd.read_csv("sales.csv", na_values=["\\N"])
df["company"] = df["company"].str.strip()
df["quarters"] = df["quarters"].str[1].astype(int)

# One row per (year, quarter, company)
base = df.groupby(
    ["years", "quarters", "company"], as_index=False
)["sales"].first()

# Add market totals as another company
total = pd.read_csv("total_sales.csv", na_values=["\\N"])
total["company"] = total["company"].str.strip()
total["quarters"] = total["quarters"].str[1].astype(int)

base = pd.concat([base, total], ignore_index=True)

# Every company for every quarter, keeping missing sales as NaN
periods = base[["years", "quarters"]].drop_duplicates()
companies = pd.DataFrame({
    "company": sorted(base["company"].unique())
})

out = periods.merge(companies, how="cross").merge(
    base, on=["years", "quarters", "company"], how="left"
)

# ---------- NEW: read company_status.csv and add zeros ----------
# Split from the right so commas inside names (e.g. "Honor Device Co., Ltd.") are safe
rows = []
with open("company_status.csv", encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, start, end, st = line.rsplit(",", 3)
        rows.append((name.strip().strip('"'), start.strip(), end.strip(), st.strip()))

status = pd.DataFrame(
    rows, columns=["company", "smartphone_start", "smartphone_end", "status"]
)
# "/N" (or "\N") -> NaN
status["smartphone_start"] = pd.to_numeric(status["smartphone_start"], errors="coerce")
status["smartphone_end"]   = pd.to_numeric(status["smartphone_end"], errors="coerce")

out = out.merge(status, on="company", how="left")

# Only NaN cells, only for companies in the status file
in_list      = out["status"].notna()
before_start = out["years"] < out["smartphone_start"]
after_end    = out["years"] > out["smartphone_end"]   # no end year -> False

mask = in_list & (before_start | after_end) & out["sales"].isna()
out.loc[mask, "sales"] = 0

# Companies with no status (expected: total row + companies not in the file)
print("No status for:", out.loc[~in_list, "company"].unique())

# Drop the helper columns: only the zeros are kept
out = out.drop(columns=["smartphone_start", "smartphone_end", "status"])
# ---------- END NEW ----------

# Create sequential quarter ID (2009 Q1 = 1)
out["id"] = (out["years"] - 2009) * 4 + out["quarters"]

out = out.sort_values(["id", "company"])[
    ["id", "years", "quarters", "company", "sales"]
]

# Save final dataset
out.to_csv("O_data.csv", index=False)

# Plot ALL companies, including total, as regular lines
pivot = out.pivot(
    index="id", columns="company", values="sales"
).sort_index()

labels = out.drop_duplicates("id").set_index("id").apply(
    lambda r: f"{r['years']} Q{r['quarters']}", axis=1
)

ax = pivot.plot(
    figsize=(16, 8),
    marker="o",
    markersize=2,
    grid=True,
    title="Smartphone sales by company"
)

ax.set_xticks(pivot.index[::4])
ax.set_xticklabels(
    labels.loc[pivot.index[::4]],
    rotation=45,
    ha="right"
)

ax.set_xlabel("Quarter")
ax.set_ylabel("Sales (million units)")
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=7
)

plt.tight_layout()
plt.show()