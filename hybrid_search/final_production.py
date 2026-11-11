from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports gigabit speeds and advanced Qos features.",
        metadata={"type": "product"}
    ),
    Document(
        page_content="For network connectivity issues, first check the ethernet cable and router status light",
        metadata={"type": "troubleshooting."}
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.",
        metadata={"type": "error"}
    ),
    Document(
        page_content="The authentication process requires valid credentials. Use 0Auth2 for secure API accesss.",
        metadata={"type": "auth"}
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel at 196.168.1.1 to modify settings.",
        metadata={"type": "config"}
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have alt text and sufficient color contrast.",
        metadata={"type": "compliance"}
    )
]



class HybridRetriever:
    """Production hybrid retriever with BM25 + Vector search"""

    def __init__(self, documents: list[Document], bm25_weight: float = 0.5, k: int = 4):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight
        self.documents = list(documents)  # keep our own copy

        #Initialize Embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        #Create vectorstore
        self.vectorstore = Chroma.from_documents(
            documents, self.embeddings, collection_name="hybrid_search"
        )

        #Create Vector retriever
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        self._build_hybrid()

    def _build_hybrid(self):
        """(Re)build BM25 and the ensemble from the current documents."""
        self.bm25_retriever = BM25Retriever.from_documents(self.documents, k=self.k)
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.bm25_weight, self.vector_weight],
        )

    def search(self, query: str) -> list[Document]:
        """Run Hybrid search using weighted RRF"""
        return self.hybrid_retriever.invoke(query)


    def add_documents(self, documents: list[Document]):
        """Add new documents to both retrievers"""
        self.documents.extend(documents)
        self.vectorstore.add_documents(documents)  # Chroma supports incremental adds
        self._build_hybrid()  # BM25 doesn't, so rebuild it and the ensemble



#Usage
retriever = HybridRetriever(documents, bm25_weight=0.5, k=4)
results = retriever.search("SKU-7742X specifications")

for doc in results:
    print(doc.page_content[:100])