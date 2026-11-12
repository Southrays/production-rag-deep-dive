import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

#Configuration
@dataclass
class Config:
    #Database - Use pooler in production
    database_url: str = os.getenv(
        "SUPABASE_DATABASE_URL",
        os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
        )
    )

    collection_name: str = "production_documents"

    #Model Settings
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    #Search settings
    default_k: int = 5
    min_similarity: float = 0.5


class RAGService:
    """Production-ready RAG service with pgvector"""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._vectorstore: PGVector | None = None
        self._chain: Runnable[str, str] | None = None


    @property
    def vectorstore(self) -> PGVector:
        """Lazy initialization of vectorstore"""

        if self._vectorstore is None:
            self._vectorstore = PGVector(
                embeddings = OpenAIEmbeddings(model=self.config.embedding_model),
                collection_name = self.config.collection_name,
                connection = self.config.database_url,
                use_jsonb = True
            )

        return self._vectorstore


    @property
    def chain(self) -> Runnable[str, str]:
        """Lazy initialization of RAG chain"""
        if self._chain is None:
            self._chain = self._create_chain()

        return self._chain


    def _create_chain(self) -> Runnable[str, str]:
        """Create the RAG chain"""
        retriever = self.vectorstore.as_retriever(
            search_kwargs = {"k": self.config.default_k}
        )

        llm = ChatOpenAI(
                    model=self.config.chat_model, temperature=self.config.temperature
                )

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant, answer the question based on the provided context.

Context:
{context}

Question:
{question}

Answer concisely and accurately. If the context doesn't contain relevant information say "I don't have enough information to answer that question."
"""
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )


    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vectorstore"""

        return self.vectorstore.add_documents(documents)


    def search(
        self, query: str, 
        k: int | None = None, 
        filter_dict: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """Search with optional filtering"""

        search_kwargs: dict[str, Any] = {"k": k or self.config.default_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return self.vectorstore.similarity_search_with_score(query, **search_kwargs)


    def ask(self, question: str) -> str:
        """Ask a question using RAG"""
        return self.chain.invoke(question)

    def ask_with_sources(self, question: str) -> dict:
        """Ask a question and return sources"""
        #Get relevant documents
        docs_with_scores = self.search(question)

        #Generate Answer
        answer = self.ask(question)

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": score
                }
                for doc, score in docs_with_scores
            ]
        }



def main():
    print("Production RAG demo")
    print("=" * 60)

    #Initialize sercice
    service = RAGService()
    print("Service ready!")

    #Add sample documents
    print("Adding sample documents")
    sample_docs = [
        Document(
            page_content="pg vector is a PostgreSQL database extension for vector similarity search.",
            metadata={"source": "docs", "topic": "pgvector"}
        ),
        Document(
            page_content="HNSW indexes provide fast approximate nearest neighbour saerch.",
            metadata={"source": "docs", "topic": "indexing"}
        ),
        Document(
            page_content="For production deployments, use connection pooling url.",
            metadata={"source": "best-practices", "topic": "production"}
        )
    ]

    ids = service.add_documents(sample_docs)
    print(f"Added {len(ids)} documents")

    print("\n Testing Search....")
    results = service.search("How do i make pgvector faster?", k=2)
    for doc, score in results:
        print(f"   Score: {score:.4f} - {doc.page_content[:50]}....")


    #Test RAG
    print("\n  Testing RAG...")
    question = "What is pgvector and how do i use it?"
    response = service.ask_with_sources(question)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources Used: {len(response['sources'])}")

    print("=" * 60)
    print("Production RAG service demo completed")
    print("=" * 60)



    if __name__ == "__main__":
        main()

