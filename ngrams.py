"""
Ngrams module for making unigram, bigram, and trigram lists for word clouds.
To Use: run in the same folder as the raw.txts
Makes (per neighbourhood): neighbourhood_unigrams.csv, neighbourhood_bigrams.csv, neighbourhood_trigrams.csv
Upload each one to wordclouds.com, clean by hand, and save cleaned versions for combiner.
"""

import glob
import re
import string
from collections import Counter
import pandas as pd
import math
from nltk.collocations import (BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder)
from nltk.tokenize import TweetTokenizer

MIN_FREQ = 10        # Atlanta used 2 for one Reddit file; raise or lower to vibe
TOP_UNIGRAMS = 300   # how many rows to export for hand-cleaning
TOP_BIGRAMS = 200
TOP_TRIGRAMS = 100

# Multi-word place names and boilerplate removed from  text before tokenising
PHRASES_TO_REMOVE = [
    "nubian square", "nubian station", "dudley square", "fort hill",
    "highland park", "jackson square", "madison park", "fields corner",
    "codman square", "uphams corner", "savin hill", "lower mills",
    "grove hall", "egleston square", "blue hill avenue", "blue hill ave",
    "dorchester avenue", "dorchester ave", "jamaica plain", "south end",
    "boston globe", "globe staff", "boston herald", "associated press",
    "can be reached at", "sign up",
]

def load_stopwords(path="stopwords.txt"):
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("#"):
                words.update(w.strip().lower() for w in line.split())
    return words

STOP = load_stopwords()
PUNCT = set(string.punctuation) | {"…", "—", "–", "''", "``"}
tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

def tokenise(text):
    text = (text.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"').lower())
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", text)   # links, emails
    for p in PHRASES_TO_REMOVE:
        text = re.sub(rf"\b{re.escape(p)}\b", " ", text)
    tokens = tknzr.tokenize(text)
    # keep real words: letters (and inner apostrophes/hyphens), 2+ characters
    return [t.strip("-'") for t in tokens
            if t not in PUNCT and re.fullmatch(r"[a-z][a-z'\-]*[a-z]", t)]

def merge_plurals(counts):
    """Fold 'evictions' into 'eviction' when both appear (Atlanta rule 4)."""
    merged = Counter(counts)
    for w in list(merged):
        if w.endswith("ies") and w[:-3] + "y" in merged:
            singular = w[:-3] + "y"
        elif w.endswith("es") and w[:-2] in merged and w[-3] in "sxz":
            singular = w[:-2]
        elif w.endswith("s") and not w.endswith("ss") and w[:-1] in merged:
            singular = w[:-1]
        else:
            continue
        merged[singular] += merged.pop(w)
    return merged

def scale(values, lo, hi):
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [hi for _ in values]
    return [round(lo + (v - vmin) * (hi - lo) / (vmax - vmin)) for v in values]

def save(rows, path):
    # same layout as the cleaned Atlanta files: weight, word, color, url
    df = pd.DataFrame(rows, columns=["weight", "word"])
    df["color"] = ""
    df["url"] = ""
    df.to_csv(path, index=False, quoting=1)

def is_stop(w):
    return w in STOP or len(w) < 3

def collocations(tokens, n, min_freq, limit, word_filter):
    """Return the most strongly associated n-grams using pointwise mutual information."""
    frequencies = Counter(tokens)
    ngram_frequencies = Counter(zip(*(tokens[i:] for i in range(n))))
    total = len(tokens)
    scored = []
    for gram, frequency in ngram_frequencies.items():
        if frequency < min_freq or word_filter(gram):
            continue
        score = math.log2(frequency * total ** (n - 1) /
                          math.prod(frequencies[word] for word in gram))
        scored.append((gram, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

files = glob.glob("*_raw.txt")
if not files:
    raise SystemExit("No *_raw.txt files here. Run scraper.py first.")

for f in sorted(files):
    n = f.replace("_raw.txt", "")
    with open(f, encoding="utf-8") as fh:
        tokens = tokenise(fh.read())
    print(f"===== {n.upper()} =====  {len(tokens):,} tokens")

    # unigrams
    counts = merge_plurals(Counter(t for t in tokens if not is_stop(t)))
    top_uni = counts.most_common(TOP_UNIGRAMS)
    save([(c, w) for w, c in top_uni], f"{n}_unigrams.csv")
    lo, hi = top_uni[-1][1], top_uni[0][1]
    print(f"  unigrams: {len(top_uni)} saved (counts {lo}-{hi})")

    # bigrams
    bi = collocations(tokens, 2, MIN_FREQ, TOP_BIGRAMS,
                      lambda gram: any(is_stop(word) for word in gram))
    if bi:
        weights = scale([s for _, s in bi], lo, hi)
        save([(wt, " ".join(g)) for (g, _), wt in zip(bi, weights)], f"{n}_bigrams.csv")
    print(f"  bigrams:  {len(bi)} saved")

    # trigrams
    tf = TrigramCollocationFinder.from_words(tokens)
    tf.apply_freq_filter(max(2, MIN_FREQ // 2))   # trigrams rarer
    tf.apply_ngram_filter(lambda a, b, c: is_stop(a) or is_stop(c))
    tri = tf.score_ngrams(TrigramAssocMeasures().pmi)[:TOP_TRIGRAMS]
    if tri:
        weights = scale([s for _, s in tri], lo, hi)
        save([(wt, " ".join(g)) for (g, _), wt in zip(tri, weights)], f"{n}_trigrams.csv")
    print(f"  trigrams: {len(tri)} saved\n")