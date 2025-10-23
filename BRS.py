import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, re
from pathlib import Path
from collections import Counter

# -------------------------
# Configuration
# -------------------------
INPUT_FILES = [r"C:\Users\DELL\Downloads\Audible_Catlog.csv",r"C:\Users\DELL\Downloads\Audible_Catlog_Advanced_Features.csv"]
    
OUTPUT_CLEAN_FILE = "cleaned_metadata.csv"
EDA_DIR = "eda_plots"

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ,]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def find_col(df, possible_names):
    """Try to match columns flexibly."""
    col_lower = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in col_lower:
            return col_lower[name.lower()]
    for c in df.columns:
        for name in possible_names:
            if name.lower() in c.lower():
                return c
    return None

dfs = []
for file in INPUT_FILES:
    if os.path.exists(file):
        df = pd.read_csv(file, low_memory=False)
        df["_source_file"] = Path(file).name
        dfs.append(df)
    else:
        print(f"⚠️ File not found: {file}")

if not dfs:
    raise FileNotFoundError("No input files found. Please ensure your CSV files are in the same folder.")

raw = pd.concat(dfs, ignore_index=True, sort=False)
print(f"\n✅ Loaded {len(raw)} rows from {len(dfs)} files.")

title_col = find_col(raw, ["title", "book_title", "name"])
author_col = find_col(raw, ["author", "authors", "book_author"])
rating_col = find_col(raw, ["rating", "avg_rating", "average_rating", "ratings"])
reviews_col = find_col(raw, ["reviews", "num_reviews", "ratings_count", "review_count"])
genres_col = find_col(raw, ["genres", "category", "categories", "genre"])
desc_col = find_col(raw, ["description", "synopsis", "summary", "about"])


df = pd.DataFrame()
df["title"] = raw[title_col].apply(clean_text) if title_col else ""
df["author"] = raw[author_col].apply(clean_text) if author_col else ""
df["description"] = raw[desc_col].fillna("").apply(clean_text) if desc_col else ""
df["genres"] = raw[genres_col].fillna("").apply(lambda s: ",".join([g.strip() for g in str(s).split(",") if g.strip()])) if genres_col else ""

df["rating"] = pd.to_numeric(raw[rating_col], errors="coerce") if rating_col else np.nan
df["num_reviews"] = pd.to_numeric(raw[reviews_col], errors="coerce").fillna(0).astype(int) if reviews_col else 0
df["_source_file"] = raw["_source_file"]

# Normalize for deduplication
df["title_norm"] = df["title"].apply(normalize_text)
df["author_norm"] = df["author"].apply(normalize_text)

before = len(df)
df = df[~((df["title"].str.len() == 0) & (df["author"].str.len() == 0))].reset_index(drop=True)
after = len(df)

df["merge_key"] = (df["title_norm"] + "||" + df["author_norm"]).apply(str.strip)
df.loc[df["author_norm"] == "", "merge_key"] = df.loc[df["author_norm"] == "", "title_norm"]

def agg_group(g):
    row = {}
    row["title"] = g["title"].iloc[0]
    row["author"] = ", ".join(sorted(set([a for a in g["author"] if a and pd.notna(a)])))
    row["description"] = max(g["description"], key=len) if any(g["description"].astype(bool)) else ""
    genres_concat = ",".join([x for xs in g["genres"] for x in xs.split(",") if x]).strip()
    seen, genres_list = set(), []
    for part in [p.strip() for p in genres_concat.split(",") if p.strip()]:
        if part.lower() not in seen:
            seen.add(part.lower())
            genres_list.append(part)
    row["genres"] = ",".join(genres_list)
    row["rating"] = g["rating"].dropna().max() if g["rating"].notna().any() else np.nan
    row["num_reviews"] = int(g["num_reviews"].sum())
    row["sources"] = ",".join(sorted(set(g["_source_file"])))
    return pd.Series(row)

cleaned = df.groupby("merge_key").apply(agg_group).reset_index(drop=True)
print(f"✅ Merged duplicates: {before} → {len(cleaned)} unique books")

os.makedirs(EDA_DIR, exist_ok=True)

# Ratings distribution
ratings = cleaned["rating"].dropna()
plt.figure(figsize=(8, 4))
plt.hist(ratings, bins=20, color="skyblue", edgecolor="black")
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.savefig(f"{EDA_DIR}/ratings_distribution.png", bbox_inches="tight")
plt.close()

all_genres = []
for g in cleaned["genres"].fillna(""):
    for part in [p.strip() for p in g.split(",") if p.strip()]:
        all_genres.append(part.lower())

from collections import Counter
genre_counts = Counter(all_genres)
top_genres = pd.DataFrame(genre_counts.most_common(20), columns=["genre", "count"])

plt.figure(figsize=(8, 5))
plt.bar(top_genres["genre"], top_genres["count"], color="lightgreen", edgecolor="black")
plt.xticks(rotation=45, ha="right")
plt.title("Top Genres (by count)")
plt.ylabel("Count")
plt.savefig(f"{EDA_DIR}/top_genres.png", bbox_inches="tight")
plt.close()

cleaned.to_csv(OUTPUT_CLEAN_FILE, index=False)
print(f"\n✅ Cleaned data saved to: {OUTPUT_CLEAN_FILE}")
print(f"📊 EDA plots saved in: {EDA_DIR}/")
print("\n--- Data Summary ---")
print(cleaned[["title", "author", "rating", "num_reviews"]].head(10))
print("\nRating stats:")
print(cleaned["rating"].describe())
print("\nTop genres:")
print(top_genres.head(10))