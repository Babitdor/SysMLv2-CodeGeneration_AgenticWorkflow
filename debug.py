# debug_approved_db.py
import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def debug_approved_solutions_db():
    """Debug the approved solutions database"""
    print("🔍 DEBUGGING APPROVED SOLUTIONS DATABASE")
    print("=" * 60)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Path to approved solutions database
    persist_dir = "./rag/approved_solutions"

    print(f"1. Checking directory: {persist_dir}")
    if os.path.exists(persist_dir):
        print(f"   ✅ Directory exists")
        contents = os.listdir(persist_dir)
        print(f"   Contents: {contents}")
    else:
        print(f"   ❌ Directory does not exist")
        return

    print(f"\n2. Initializing ChromaDB...")
    try:
        db = Chroma(
            collection_name="approved_solutions",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        print("   ✅ ChromaDB initialized")
    except Exception as e:
        print(f"   ❌ Error initializing ChromaDB: {e}")
        return

    print(f"\n3. Checking collection...")
    try:
        collection = db._collection
        count = collection.count()
        print(f"   Collection count: {count}")

        # Try to get all documents
        if count > 0:
            all_docs = collection.get()
            print(f"   Document IDs: {all_docs.get('ids', [])}")
            print(f"   Metadata count: {len(all_docs.get('metadatas', []))}") # type: ignore
        else:
            print("   No documents in collection")

    except Exception as e:
        print(f"   ❌ Error checking collection: {e}")

    print(f"\n4. Testing similarity search...")
    try:
        results = db.similarity_search("test", k=5)
        print(f"   Similarity search found: {len(results)} results")
        for i, doc in enumerate(results):
            print(f"   {i+1}. Content: {doc.page_content[:50]}...")
            print(f"      Metadata: {doc.metadata}")
    except Exception as e:
        print(f"   ❌ Error in similarity search: {e}")

    print(f"\n5. Checking for full_entries directory...")
    full_entries_dir = os.path.join(persist_dir, "full_entries")
    if os.path.exists(full_entries_dir):
        json_files = [f for f in os.listdir(full_entries_dir) if f.endswith(".json")]
        print(f"   Found {len(json_files)} JSON files in full_entries:")
        for f in json_files[:5]:  # Show first 5
            print(f"   - {f}")
    else:
        print("   ❌ full_entries directory not found")


if __name__ == "__main__":
    debug_approved_solutions_db()
