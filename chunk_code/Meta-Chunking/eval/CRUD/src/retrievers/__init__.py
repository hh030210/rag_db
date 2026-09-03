from .base import BaseRetriever

# The formal denoise ablation uses BaseRetriever only. Keep optional BM25 and
# reranker implementations importable when their extra dependencies exist,
# without making the Milvus-only path require Elasticsearch/FlagEmbedding.
try:
    from .bm25 import CustomBM25Retriever
except ImportError:
    CustomBM25Retriever = None

try:
    from .hybrid import EnsembleRetriever
except ImportError:
    EnsembleRetriever = None

try:
    from .hybrid_rerank import EnsembleRerankRetriever
except ImportError:
    EnsembleRerankRetriever = None
