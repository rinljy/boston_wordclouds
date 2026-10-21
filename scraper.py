""""
Scraper module for filtering and scraping full article text from Media Cloud exports.
To Use: place Media Cloud exports into a folder called "mediacloud_exports", with individual export files simply being the name of the neighbourhood.
Makes (per neighbourhood): neighbourhood_articles.csv, with one row per article (title, source, date, full text)
Also will me a backup of everything that's scraped, csv of dropped_urls, and articles removed as off-topic.
If this stops partway, simply run again and will pick up where it left off.
"""

import glob
import os
import re
import time
import pandas as pd

INPUT_FOLDER = "mediacloud_exports"
SCRAPED_FILE = "articles_scraped.csv"
MIN_WORDS = 100 # Minimum number of words for an article to be considered valid (avoid paywalls and videos etc.)
PAUSE_SECONDS = 1 # Pause between scraping articles to avoid rate limiting
SAVE_EVERY = 25

# List of Boston outlets from which articles in the CSV are kept (list obtained from generative AI)
BOSTON_OUTLETS = {
    "bostonglobe.com", "boston.com", "bostonherald.com", "boston.gov",
    "nbcboston.com", "thebostonpilot.com", "thebostonsun.com",
    "boston25news.com", "bostonirish.com", "northbostonnews.com",
    "bostonspiritmagazine.com", "newbostonpost.com", "wbur.org",
    "baystatebanner.com", "dotnews.com", "universalhub.com", "wgbh.org",
    "wcvb.com",
}

# Other outlets: kept only if the headline clearly points to Boston
BOSTON_HEADLINE = re.compile(
    r"\b(boston|roxbury|mattapan|massachusetts|nubian square|dudley square|"
    r"codman square|fields corner|uphams corner|savin hill|ashmont|grove hall|"
    r"egleston|blue hill ave|dorchester ave)\b", re.IGNORECASE)
NOT_BOSTON_DORCHESTER = re.compile(
    r"dorchester county|summerville|charleston|dorset", re.IGNORECASE)
NOT_BOSTON_OUTLETS = {"postandcourier.com"}

def keep_row(row):
    src = str(row.get("media_name", "")).lower()
    title = str(row.get("title", ""))
    if src in NOT_BOSTON_OUTLETS:
        return bool(BOSTON_HEADLINE.search(title))
    if src in BOSTON_OUTLETS:
        return True
    if BOSTON_HEADLINE.search(title):
        return True
    if re.search(r"\bdorchester\b", title, re.I) and not NOT_BOSTON_DORCHESTER.search(title):
        return True
    return False


# Read and Filter articles
files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
if not files:
    raise SystemExit(f"No CSV files in '{INPUT_FOLDER}'. Make the folder and put your exports in it.")

frames = []
for f in files:
    df = pd.read_csv(f)
    df["neighborhood"] = os.path.basename(f).split("-")[0].split(".")[0].strip().lower()
    frames.append(df)
allrows = pd.concat(frames, ignore_index=True)

if "language" in allrows.columns:
    allrows = allrows[allrows["language"].fillna("en") == "en"]

mask = allrows.apply(keep_row, axis=1)
allrows[~mask].to_csv("dropped_urls.csv", index=False)
kept = allrows[mask]
print(f"Read {len(allrows)} rows; kept {mask.sum()}, dropped {(~mask).sum()} as off-topic")

# Have one row per URL
per_url = kept.groupby("url", as_index=False).agg(
    title=("title", "first"),
    media_name=("media_name", "first") if "media_name" in kept else ("url", "first"),
    publish_date=("publish_date", "first") if "publish_date" in kept else ("url", "first"),
    neighborhoods=("neighborhood", lambda s: "; ".join(sorted(set(s)))),
)
print(f"{len(per_url)} unique articles ({len(kept) - len(per_url)} duplicates across searches)\n")

# scrape
from newspaper import Article

done = pd.read_csv(SCRAPED_FILE) if os.path.exists(SCRAPED_FILE) else pd.DataFrame(columns=["url"])
results = done.to_dict("records")
todo = per_url[~per_url["url"].isin(set(done["url"]))]
print(f"{len(done)} already scraped, {len(todo)} to go "
        f"(about {len(todo) * 2 // 60} minutes)\n")

for i, row in enumerate(todo.itertuples(index=False), start=1):
    rec = {"url": row.url}
    try:
        a = Article(row.url)
        a.download()
        a.parse()
        rec["text"], rec["error"] = a.text, ""
    except Exception as e:
        rec["text"], rec["error"] = "", str(e)[:200]
    results.append(rec)
    print(f"[{i}/{len(todo)}] {'ok' if not rec['error'] else 'FAILED'}  {row.url[:80]}")
    if i % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(SCRAPED_FILE, index=False)
    time.sleep(PAUSE_SECONDS)
pd.DataFrame(results).to_csv(SCRAPED_FILE, index=False)

texts = pd.DataFrame(results).drop_duplicates("url").set_index("url")
per_url["text"] = per_url["url"].map(texts["text"]).fillna("")
per_url["error"] = per_url["url"].map(texts["error"]).fillna("")
scraped = per_url

scraped["word_count"] = scraped["text"].fillna("").str.split().str.len()
scraped["also_in_other_neighborhood"] = scraped["neighborhoods"].str.contains(";")

# one file per neighbourhood
print()
for n in sorted(kept["neighborhood"].unique()):
    part = scraped[scraped["neighborhoods"].str.split("; ").apply(lambda l: n in l)]
    part.drop(columns=["neighborhoods"]).to_csv(f"{n}_articles.csv", index=False)
    full = part[part["word_count"] >= MIN_WORDS]
    with open(f"{n}_raw.txt", "w", encoding="utf-8") as out:
        out.write("\n\n".join(full["text"]))
    print(f"{n}: {len(part)} articles, {len(full)} with full text "
          f"-> {n}_articles.csv, {n}_raw.txt")