import chromadb

chroma_client = chromadb.Client()

collection_name = "test-collection"

collection = chroma_client.get_or_create_collection(collection_name)

documents = [
    {"id": "doc1", "text": "Hello World"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later"}
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=doc["text"])


query_text = "Hello World"

results = collection.query(
    query_texts=[query_text],
    n_results = 3
    )

print(results)