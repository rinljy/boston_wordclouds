""""
Combiner module combines ngrams into one word file per neighbourhood
To Use: save cleaned files from wordclouds.com in this folder as neighbourhood_ngram_clean.csv and run script
Makes (per neighbourhood): neighbourhood_final_csv -> upload to wordclouds.com
Rule 5 is applied but can be turned off (keep highest level of word)
"""

import glob
import os
import pandas as pd

APPLY_RULE_5 = True

def read(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["weight", "word"])
    df = pd.read_csv(path, sep=None, engine="python")   # handles , or ;
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["weight", "word"]].dropna()
    df["word"] = df["word"].astype(str).str.strip().str.lower()
    return df

neighbourhoods = sorted({os.path.basename(f).split("_")[0]
                        for f in glob.glob("*_clean.csv")})
if not neighbourhoods:
    raise SystemExit("No *_clean.csv files found. Save cleaned lists first.")

for n in neighbourhoods:
    uni = read(f"{n}_unigrams_clean.csv")
    bi = read(f"{n}_bigrams_clean.csv")
    tri = read(f"{n}_trigrams_clean.csv")
    print(f"===== {n.upper()} =====  {len(uni)} unigrams, {len(bi)} bigrams, {len(tri)} trigrams")

    if APPLY_RULE_5:
        tri_words = [set(t.split()) for t in tri["word"]]
        tri_text = [" " + t + " " for t in tri["word"]]
        bi_words = [set(b.split()) for b in bi["word"]]

        drop_bi = bi["word"].apply(lambda b: any(" " + b + " " in t for t in tri_text))
        drop_uni = uni["word"].apply(
            lambda u: any(u in s for s in bi_words + tri_words))

        if drop_uni.any():
            print("  unigrams dropped (inside a bigram/trigram):",
                  ", ".join(uni.loc[drop_uni, "word"]))
        if drop_bi.any():
            print("  bigrams dropped (inside a trigram):",
                  ", ".join(bi.loc[drop_bi, "word"]))
        uni, bi = uni[~drop_uni], bi[~drop_bi]

    final = (pd.concat([uni, bi, tri])
               .sort_values("weight", ascending=False)
               .drop_duplicates("word"))
    final["weight"] = final["weight"].round().astype(int)
    final["color"] = ""
    final["url"] = ""
    final[["weight", "word", "color", "url"]].to_csv(
        f"{n}_wordcloud_final.csv", index=False, quoting=1)
    print(f"  saved {n}_wordcloud_final.csv ({len(final)} entries)\n")