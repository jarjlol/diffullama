"""Shared utilities: HTTP with retry, abstract reconstruction, pure-python TF-IDF."""
import json
import math
import re
import time
import urllib.parse
from collections import Counter

import requests

OPENALEX = "https://api.openalex.org"
HEADERS = {"User-Agent": "litreview-coursework-pipeline/1.0 (mailto:research-project@example.com)"}


def get(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code >= 500:
            time.sleep(2 * (attempt + 1))
            continue
        # 4xx other than 429: don't retry
        return None
    return None


def reconstruct_abstract(inv_index):
    if not inv_index:
        return ""
    positions = {}
    max_pos = 0
    for word, idxs in inv_index.items():
        for i in idxs:
            positions[i] = word
            max_pos = max(max_pos, i)
    return " ".join(positions.get(i, "") for i in range(max_pos + 1))


STOPWORDS = set("""a an the of and or to in on for with is are was were be been being this that these those
by as at from into over under between within without across per via using use used based based-on we our
their its it he she they i you your paper propose proposed method models model results show shows study
studies work works approach approaches novel new recent recently large language llm llms text data
""".split())


def tokenize(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def tfidf_vectors(docs):
    """docs: list[str] -> list[Counter] of tf-idf weighted term vectors (pure python)."""
    tokenized = [tokenize(d) for d in docs]
    df = Counter()
    for toks in tokenized:
        for term in set(toks):
            df[term] += 1
    n = len(docs)
    vectors = []
    for toks in tokenized:
        tf = Counter(toks)
        vec = {}
        for term, count in tf.items():
            idf = math.log((n + 1) / (df[term] + 1)) + 1
            vec[term] = (1 + math.log(count)) * idf
        vectors.append(vec)
    return vectors


def cosine(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def percentile_rank(values):
    """Return percentile rank (0-1, higher=better) for each value in the list, ties averaged."""
    n = len(values)
    if n <= 1:
        return [1.0] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank / (n - 1) if n > 1 else 1.0
        i = j + 1
    return ranks


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path) as f:
        return json.load(f)
