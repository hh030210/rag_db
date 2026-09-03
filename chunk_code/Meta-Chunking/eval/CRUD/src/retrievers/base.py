from abc import ABC
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.node_parser import SimpleNodeParser
from llama_index import download_loader

from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext, StorageContext
try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    from langchain.schema.embeddings import Embeddings
from llama_index.vector_stores import MilvusVectorStore
import os
from llama_index.data_structs import Node
from pymilvus import connections

def _milvus_uri():
    """Return a remote Milvus URI or a local Milvus Lite database path."""
    return os.environ.get("DENOISE_MILVUS_URI", "http://localhost:19530")


_LOCAL_MILVUS_GRPC_OPTIONS = {
    # Milvus Lite 的本地 unix socket 在长时间向量化期间没有 RPC，
    # 不应每 10 秒发送一次空闲 keepalive，否则服务端会返回 Too many pings。
    "grpc.keepalive_time_ms": 600000,
    "grpc.keepalive_timeout_ms": 20000,
    "grpc.keepalive_permit_without_calls": False,
}


def _patch_local_milvus_client_keepalive():
    """Pass safe gRPC keepalive options through LlamaIndex's MilvusClient call."""
    try:
        from pymilvus.milvus_client import MilvusClient
    except ImportError:
        return

    if getattr(MilvusClient, "_denoise_keepalive_patched", False):
        return

    original_init = MilvusClient.__init__

    def _patched_init(self, *args, **kwargs):
        uri = kwargs.get("uri", args[0] if args else "http://localhost:19530")
        if isinstance(uri, str) and not uri.startswith(("http://", "https://")):
            kwargs.setdefault("grpc_options", _LOCAL_MILVUS_GRPC_OPTIONS)
        original_init(self, *args, **kwargs)

    MilvusClient.__init__ = _patched_init
    MilvusClient._denoise_keepalive_patched = True


def _ensure_milvus_connection():
    """Ensure a Milvus connection alias 'default' exists for llama_index MilvusVectorStore.

    llama_index 的 MilvusVectorStore 内部会创建临时 alias（cm-XXXX），
    然后用 ORM 的 Collection(... using=alias) 去查，但 ORM 端并没有这个 alias。
    我们在调用 MilvusVectorStore 之前 patch 一下 pymilvus 的 _fetch_handler，
    使它在任何 alias 上都返回 default 的 handler。
    """
    _patch_local_milvus_client_keepalive()
    try:
        if not connections.has_connection("default"):
            connections.connect(
                alias="default",
                uri=_milvus_uri(),
                grpc_options=_LOCAL_MILVUS_GRPC_OPTIONS,
            )
    except Exception:
        pass

    # Patch: 让 pymilvus ORM 的 _fetch_handler 兜底返回 default 的 handler
    try:
        from pymilvus.orm import connections as _conn_mod
        _orig_fetch = _conn_mod.Connections._fetch_handler
        def _patched_fetch(self, using):
            try:
                return _orig_fetch(self, using)
            except Exception:
                return _orig_fetch(self, "default")
        _conn_mod.Connections._fetch_handler = _patched_fetch
    except Exception:
        pass


class BaseRetriever(ABC):
    def __init__(
            self,
            docs_directory: str,
            embed_model: Embeddings,
            embed_dim: int = 768,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs",
            construct_index: bool = False,
            add_index: bool = False,
            similarity_top_k: int=2,
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k

        # 关键修复:无论是构造索引还是加载索引,都要先把 langchain Embeddings
        # 包成 llama_index 的 LangchainEmbedding,这样 query 阶段才能调
        # get_agg_embedding_from_queries / _get_query_embedding 等方法。
        if not isinstance(self.embed_model, LangchainEmbedding):
            self.embed_model = LangchainEmbedding(self.embed_model)

        if construct_index:
            self.construct_index()
        else:
            self.load_index_from_milvus()

        if add_index:
            self.add_index()

        # 关键:把 vector_retriever 存为 self 属性,search_docs 直接用,
        # 避免走 RetrieverQueryEngine 触发 LLM 合成与 context size 检查。
        self.vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=self.similarity_top_k,
        )

    def construct_index(self):
        folder_path = self.docs_directory
        _ensure_milvus_connection()
        nodes=[]
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                relative_path = os.path.join(root, file)
                with open(relative_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                aa=content.split('\n')
                for i in aa:
                    if len(i)<10:
                        continue
                    node1 = Node(text=i)
                    nodes.append(node1)

        if len(nodes) == 0:
            raise ValueError(f"没有读到任何 chunk，请检查 docs 目录: {folder_path}")
        print(f"[Index] 共读取 {len(nodes)} 个节点，开始向量化...")

        if not isinstance(self.embed_model, LangchainEmbedding):
            self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        vector_store = MilvusVectorStore(
            uri=_milvus_uri(), token="",
            dim=self.embed_dim, overwrite=True,
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Process and index nodes in chunks due to Milvus limitations
        for spilt_ids in range(0, len(nodes), 8000):
            end_ids = min(spilt_ids + 8000, len(nodes))
            batch_nodes = nodes[spilt_ids:end_ids]
            texts_batch = [n.get_text() for n in batch_nodes]

            # Pre-compute embeddings with batch for speed
            embeddings_batch = self.embed_model.get_text_embedding_batch(texts_batch)
            for node, emb in zip(batch_nodes, embeddings_batch):
                node.embedding = emb

            self.vector_index = GPTVectorStoreIndex(
                batch_nodes, service_context=service_context,
                storage_context=storage_context, show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")

            vector_store = MilvusVectorStore(
                uri=_milvus_uri(), token="",
                overwrite=False,
                collection_name=self.collection_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("Indexing finished!")

    def add_index(self):  #一般没有用到？
        _ensure_milvus_connection()
        if self.docs_type == 'json':
            JSONReader = download_loader("JSONReader")
            documents = JSONReader().load_data(self.docs_directory)
        else:
            documents = SimpleDirectoryReader(self.docs_directory).load_data()

        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

        if not isinstance(self.embed_model, LangchainEmbedding):
            self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        vector_store = MilvusVectorStore(
            uri=_milvus_uri(), token="",
            overwrite=False,
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

         # Process and index nodes in chunks due to Milvus limitations
        for spilt_ids in range(0, len(nodes), 8000):
            self.vector_index = GPTVectorStoreIndex(
                nodes[spilt_ids:spilt_ids+8000], service_context=service_context,
                storage_context=storage_context, show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")

        print("Indexing finished!")

    def load_index_from_milvus(self):
        _ensure_milvus_connection()
        vector_store =  MilvusVectorStore(
            uri=_milvus_uri(), token="",
            overwrite=False, dim=self.embed_dim,
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm=None)
        self.vector_index = GPTVectorStoreIndex(
            [], storage_context=storage_context,
            service_context=service_context,
        )

    def search_docs(self, query_text: str):
        # 关键修复:不走 RetrieverQueryEngine(它会内部调 LLM,触发
        # 'available context size' 检查,以及不必要的 response 合成)。
        # 直接用 self.vector_retriever.retrieve() 拿 source_nodes 即可。
        source_nodes = self.vector_retriever.retrieve(query_text)
        chunks_text = []
        for sn in source_nodes:
            if hasattr(sn, 'node') and sn.node and sn.node.text:
                chunks_text.append(sn.node.text)
        return "\n\n".join(chunks_text)

