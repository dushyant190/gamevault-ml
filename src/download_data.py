import duckdb
import pandas as pd

url = "https://huggingface.co/datasets/atalaydenknalbant/video-games-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

print("Reading game data...")

query = f"""
SELECT *
FROM read_parquet('{url}')
LIMIT 1000
"""

df = duckdb.sql(query).df()

print("Finished!")
print("Rows:", len(df))
print("Columns:")
print(df.columns.tolist())

df.to_csv("data/games.csv", index=False)

print("Saved to data/games.csv")