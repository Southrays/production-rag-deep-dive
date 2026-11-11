import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def basic_embeddings():
    #Single text
    text = "What is Machine Learning?"
    single_embedding = embeddings.embed_query(text)

    print(f"Vector dimension: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")

    #Normalize the vectors so we can see them better
    #Normalizing vectors prevents longer documents from having bigger vectors
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")



def batch_embeddings():
    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    batch_embeddings = embeddings.embed_documents(text)

    for i, emb in enumerate(batch_embeddings):
        print(f"Text {i + 1} - Vector dimension: {len(emb)}")
        print(f"Text {i + 1} - First 5 values: {emb[:5]}")
        print(f"Text {i + 1} - Vector norm: {np.linalg.norm(emb):.4f}")



def similarity_search():

    # Documents
    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist"

    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1)) * (np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    ranked_docs = sorted(zip(docs, similarities), key=lambda x:[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarities")
    for doc, score in ranked_docs:
        print(f"{score:.4f}: {doc}")
