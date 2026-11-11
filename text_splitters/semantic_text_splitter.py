from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

document = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
"""


test_queries = [
    "What is machine learning?",
    "What is unsupervised learning?",
    "Which fields use machine learning?"
]


recursive_splitter = RecursiveCharacterTextSplitter(
    chunks=400, 
    overlap=50, 
    separators=["\n\n", "\n", ". ", " " ]
)

recursive_chunks = recursive_splitter.split_text(document)

print(f"Recursive chunks: {len(recursive_chunks)}")
for i, chunk in enumerate(recursive_chunks):
    print(f"\\n---- Chunk {i+1} ({len(chunk)} chars) ----")
    print(chunk[:100] + "...." if len(chunk) > 100 else chunk )


recursive_vectorstore = Chroma.from_texts(
    recursive_chunks,
    embeddings,
    collection_name="recursive_chunks"
)





semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90  #Break at the 90th percentile dissimilarity
)

semantic_chunks = semantic_chunker.split_text(document)

print(f"Semantic chunks: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks):
    print(f"\\n---- Chunk {i+1} ({len(chunk)} chars) ----")
    print(chunk[:100] + "...." if len(chunk) > 100 else chunk )

semantic_vectorstore = Chroma.from_texts(
    semantic_chunks,
    embeddings,
    collection_name="semantic_chunks"
)





def test_retrieval(query: str, vectorstore, name: str):
    results = vectorstore.similarity_search(query, k=1)
    print(f'\\n{name} - Query: \"{query}\"')
    print(f"Retrieved: {results[0].page_content[:150]}...")
    return results[0].page_content


print(f"\n{"=" * 60}")
print(" RETRIEVAL TEST")
print("=" * 60)


for query in test_queries:
    print("=" * 60)
    recursive_result = test_retrieval(query, recursive_vectorstore, "RECURSIVE")
    semantic_result = test_retrieval(query, semantic_vectorstore, "SEMANTIC")



def smart_chunker(
    text: str, 
    use_semantic: bool = True, 
    fallback_chunk_size: int = 500
) -> list[str]:
    """Production chunking with semantic as primary and recursive as fallback."""

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90
            )
            chunks = chunker.split_text(text)

            #Validate chunks aren't too large
            max_chunk_size = 2000
            if any(len(c) > max_chunk_size for c in chunks):
                #fallback to recursive for oversized chunks
                return _recursive_fallback(text, fallback_chunk_size)

            return chunks
        
        except (TypeError, ValueError) as e:
            print(f"Semantic chunking failed: {e}, using fallback")
            return _recursive_fallback(text, fallback_chunk_size)

    return _recursive_fallback(text, fallback_chunk_size)


def _recursive_fallback(text, chunk_size) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50
        )

        return splitter.split_text(text)



chunks = smart_chunker(document, use_semantic=True)
print(f"Created {len(chunks)} semantic chunks")
