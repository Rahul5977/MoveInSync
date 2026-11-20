"""Lexical retrieval for hybrid mode (D10, D55, D56). Setup fixed before any results (D55):

- lowercase, tokens = regex `\\w+`; English stopwords (NLTK list) removed, then Snowball stemming
- index exactly the text that is embedded (context header + body)
- Okapi BM25, k1 = 1.5, b = 0.75, idf = ln(1 + (N - n + 0.5) / (n + 0.5)); only score > 0 returned

One in-memory index per collection name. Names are versioned (`nimbus_v2`), so a new version is a
new index; the cache is looked up through the active-version pointer on every request (D56).
"""

import math
import re
import threading
from collections import Counter

import snowballstemmer

from app.schemas import RetrievedChunk
from app.store.vector import VectorStore

K1, B = 1.5, 0.75

# NLTK English stopword list (179 words), embedded so there is no runtime download.
STOPWORDS = frozenset(
    """i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself
    yourselves he him his himself she she's her hers herself it it's its itself they them their
    theirs themselves what which who whom this that that'll these those am is are was were be been
    being have has had having do does did doing a an the and but if or because as until while of
    at by for with about against between into through during before after above below to from up
    down in out on off over under again further then once here there when where why how all any
    both each few more most other some such no nor not only own same so than too very s t can will
    just don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn
    didn't doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn
    mustn't needn needn't shan shan't shouldn shouldn't wasn wasn't weren weren't won won't
    wouldn wouldn't""".split()
)

_TOKEN_RE = re.compile(r"\w+")
_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]
    return _stemmer.stemWords(words)


class Bm25Index:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        n_docs = len(chunks)
        self._avg = (sum(self._len) / n_docs) if n_docs else 0.0
        df = Counter(term for tf in self._tf for term in tf)
        self._idf = {t: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        terms = [t for t in tokenize(query) if t in self._idf]
        scored = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = K1 * (1 - B + B * self._len[i] / self._avg)
                    score += self._idf[t] * f * (K1 + 1) / (f + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], self.chunks[x[1]].chunk_id))
        return [self.chunks[i].model_copy(update={"score": s}) for s, i in scored[:k]]


class Bm25Cache:
    """collection_name -> Bm25Index. Built lazily or on activation; thread-safe."""

    def __init__(self) -> None:
        self._indexes: dict[str, Bm25Index] = {}
        self._lock = threading.Lock()

    def build(self, store: VectorStore, name: str) -> Bm25Index:
        index = Bm25Index(store.all_chunks(name))
        with self._lock:
            # A company has one live collection; drop its older indexes (same "{co}_v" prefix).
            prefix = name.rsplit("_v", 1)[0] + "_v"
            for old in [n for n in self._indexes if n.startswith(prefix) and n != name]:
                del self._indexes[old]
            self._indexes[name] = index
        return index

    def get(self, store: VectorStore, name: str) -> Bm25Index:
        with self._lock:
            index = self._indexes.get(name)
        return index if index is not None else self.build(store, name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._indexes)
