import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

#For demo, use local if Supabase is not configured
DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)


def connect_to_supabase():
    """Connect to Supabase pgvector"""

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    #Note: Supabase has pgvector pre-installed
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=DATABASE_URL,
        use_jsonb=True
    )

    return vectorstore


def verify_connection(vectorstore):
    """Verify the connection works"""

    #Add a test document
    test_doc = Document(
        page_content="This is a document to verify Supabase connection",
        metadata={"test": True}
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"Added test document: {ids[0]}")

        #Search for it
        results = vectorstore.similarity_search("test document")
        if results:
            print(f"Search works: {results[0].page_content}")

        #Clean up
        vectorstore.delete(ids)
        print("Cleanup complete")

        return True

    except (ValueError, TypeError) as e:
        print(f"Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Supabase pgvector Connection Test")
    print("=" * 60)


    if SUPABASE_URL:
        print("Connecting to Supabase")
        print(f"Host: {SUPABASE_URL.split("@")}")

    else:
        print("\n SUPABASE_DATABASE_URL not set")
        print("  Using local POSTGRESQL instead")
        print("\n  To use Supaase")