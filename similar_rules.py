"""Selection rules for the ReadWell book recommender.
 
1. Do not recommend the selected book itself.
2. Exclude boxed sets, collections and omnibus editions.
3. Exclude books from the same series.
4. Reduce the similarity score of books by the same author to 40%.
5. Include no more than one book per author and series.
6. Give priority to books that have review data.
"""
import re

import numpy as np

BUNDLE_WORDS = r"box(?:ed)? set|collection|complete series|omnibus|books? \d|\d+-book|#\d+\s*-\s*\d+|#\d+-\d+"
AUTHOR_PENALTY = 0.40  # a same-author book keeps 40% of its similarity score
POOL_FACTOR = 3        # n * POOL_FACTOR good candidates are collected before the review-data-first step

_SERIES_IN_TITLE = re.compile(r"\(([^()#]+?),?\s*#\s*[\d.]+[^()]*\)")


def _norm_series_name(name):
    name = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower())
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"^(the|a|an) ", "", name)
    return name


def series_keys(series_ids, title):
    """Return the keys that tell which series a book belongs to.
 
    The keys come from two places: the series ids in the dataset, and the series
    name written in the title, e.g. 'Catching Fire (The Hunger Games, #2)'.
    """
    keys = set()
    for sid in series_ids or []:
        sid = str(sid).strip()
        if sid:
            keys.add("id:" + sid)
    m = _SERIES_IN_TITLE.search(str(title))
    if m:
        name = _norm_series_name(m.group(1))
        if name:
            keys.add("name:" + name)
    return frozenset(keys)


def bundle_mask(titles):
    """True for titles that look like boxed sets, collections or omnibus editions."""
    return titles.str.contains(BUNDLE_WORDS, case=False, regex=True).to_numpy()


def relation_masks(index, author_ids, series_sets):
    """For the chosen book: which books are in its series, and which have the same author."""
    n = len(author_ids)
    query_series = series_sets[index]
    query_author = author_ids[index]
    same_series = np.zeros(n, dtype=bool)
    if query_series:
        same_series = np.array([bool(s & query_series) for s in series_sets], dtype=bool)
    same_author = np.zeros(n, dtype=bool)
    if query_author:
        same_author = np.array([a == query_author for a in author_ids], dtype=bool)
    return same_series, same_author


def pick_similar(scores, index, author_ids, series_sets, is_bundle, has_reviews, n=5,
                 same_series=None, same_author=None,
                 author_penalty=AUTHOR_PENALTY, pool_factor=POOL_FACTOR):
    """Choose n books to recommend for the book at position `index`.
 
    scores       how similar every book is to the chosen book (cosine similarity, 0 to 1)
    author_ids   author id of every book (empty if unknown)
    series_sets  series keys of every book, from series_keys()
    is_bundle    True for boxed sets and collections
    has_reviews  True if the book has review data
    same_series / same_author  optional ready-made yes/no lists (made here if left out)
 
    Returns the positions of n books (fewer if there are not enough good ones).
    """
    if same_series is None or same_author is None:
        same_series, same_author = relation_masks(index, author_ids, series_sets)

    s = np.array(scores, dtype=float).copy()
    s[index] = -1.0                      # rule 1
    s[is_bundle] = -1.0                  # rule 2
    s[same_series] = -1.0                # rule 3
    penalised = same_author & (s > 0)    # rule 4
    s[penalised] *= author_penalty

    order = np.argsort(-s, kind="stable")
    limit = n * pool_factor
    pool, used_series, used_authors = [], set(), set()
    for i in order:
        if s[i] < 0:
            break
        if series_sets[i] & used_series:   # rule 5
            continue
        if author_ids[i] and author_ids[i] in used_authors:
            continue
        pool.append(int(i))
        used_series |= series_sets[i]
        if author_ids[i]:
            used_authors.add(author_ids[i])
        if len(pool) >= limit:
            break

    # rule 6: books with review data go first; inside each group the order stays the same
    pool = [i for i in pool if has_reviews[i]] + [i for i in pool if not has_reviews[i]]
    return pool[:n]
