import pandas as pd

DONORS = ["AMD", "INTC", "QCOM", "AVGO", "MU", "TXN", "ADI", "MCHP", "MRVL", "NXPI"]

df = pd.read_parquet("data/features/daily_features.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df[df["ticker"].isin(DONORS)]
df["year"] = df["date"].dt.year

by_year = df.groupby("year").agg(
    log_return_std=("log_return", "std"),
    log_high_low_range_mean=("log_high_low_range", "mean"),
)
print(by_year.loc[2012:2022])
print("\n¿2022 es el año más volátil de 2012-2022? ->", by_year.loc[2012:2022, "log_return_std"].idxmax() == 2022)