from typing import Optional, List, Dict, Any
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from States import ApprovedCodeEntry
import os

class KnowledgeBase:
    """Vector database storage for approved SysML code entries"""

    def __init__(
        self,
        db_path: str = "./sysml_knowledge_base",
        collection_name: str = "approved_sysml_codes",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Initialize components
        self.client = None
        self.collection = None
        self.embedding_model = None

        self._initialize_storage()
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

    def _initialize_storage(self):
        """Initialize the vector database and embedding model"""
        try:
            # Initialize ChromaDB client
            print(f"🔧 Initializing vector database at {self.db_path}")
            settings = Settings(
                anonymized_telemetry=False, allow_reset=True, is_persistent=True
            )
            self.client = chromadb.PersistentClient(
                path=self.db_path, settings=settings
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Human-approved SysML code entries",
                    "hnsw_space": "cosine",
                },
            )

            # Initialize embedding model
            print(f"🤖 Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            print("✅ Vector database initialized successfully!")
            print(f"📊 Current collection size: {self.collection.count()}")

        except Exception as e:
            print(f"❌ Failed to initialize vector database: {e}")
            raise

    def store_approved_entry(self, entry: ApprovedCodeEntry) -> bool:
        """Store an approved code entry in the vector database"""
        try:
            print(f"💾 Storing approved code entry: {entry.id}")

            # Generate embedding for the entry
            embedding = self.embedding_model.encode(entry.embedding_text).tolist()  # type: ignore

            # Prepare metadata (ChromaDB has limitations on nested objects)
            metadata = {
                "task": entry.task[:500],  # Truncate for metadata limits
                "created_at": entry.created_at,
                "code_length": len(entry.generated_code),
                "validation_success_rate": entry.validation_info.get(
                    "success_rate", 0.0
                ),
                "iterations_used": entry.workflow_metadata.get("iterations_used", 0),
                "has_human_feedback": bool(entry.human_feedback),
                "workflow_status": entry.workflow_metadata.get("workflow_status", ""),
            }

            # Store in vector database
            self.collection.add(  # type: ignore
                ids=[entry.id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[
                    entry.embedding_text
                ],  # Store the embedding text as document
            )

            # Also store full entry as JSON for complete retrieval
            self._store_full_entry_json(entry)

            print(f"✅ Successfully stored entry {entry.id}")
            print(f"📈 Collection size now: {self.collection.count()}")  # type: ignore
            return True

        except Exception as e:
            print(f"❌ Failed to store entry {entry.id}: {e}")
            return False

    def _store_full_entry_json(self, entry: ApprovedCodeEntry):
        """Store the complete entry as JSON for full retrieval"""
        import os

        # Create JSON storage directory
        json_dir = os.path.join(self.db_path, "full_entries")
        os.makedirs(json_dir, exist_ok=True)

        # Save full entry
        json_path = os.path.join(json_dir, f"{entry.id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score (0-1 range)"""
        # For cosine distance (most common)
        if distance <= 1.0:
            return max(0.0, 1 - distance)
        # For euclidean or other metrics
        else:
            return max(0.0, 1 / (1 + distance))

    def search_similar_entries(
        self, query: str, n_results: int = 5, include_code: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for similar entries based on query"""
        try:
            print(f"🔍 Searching for similar entries: '{query[:50]}...'")

            # Generate embedding for query
            query_embedding = self.embedding_model.encode(query).tolist()  # type: ignore

            # Search in vector database
            results = self.collection.query(  # type: ignore
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
            )

            # Process results
            similar_entries = []
            for i in range(len(results["ids"][0])):
                entry_id = results["ids"][0][i]
                distance = results["distances"][0][i]  # type: ignore
                metadata = results["metadatas"][0][i]  # type: ignore
                document = results["documents"][0][i]  # type: ignore

                result_entry = {
                    "id": entry_id,
                    "similarity_score": self._distance_to_similarity(
                        distance
                    ),  # Convert distance to similarity
                    "task": metadata["task"],
                    "created_at": metadata["created_at"],
                    "metadata": metadata,
                    "embedding_text": document,
                }

                # Optionally include full code
                if include_code:
                    full_entry = self._load_full_entry_json(entry_id)
                    if full_entry:
                        result_entry["full_entry"] = full_entry

                similar_entries.append(result_entry)

            print(f"✅ Found {len(similar_entries)} similar entries")
            return similar_entries

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def _load_full_entry_json(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Load the complete entry from JSON storage"""
        import os

        json_path = os.path.join(self.db_path, "full_entries", f"{entry_id}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load full entry {entry_id}: {e}")
        return None

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored entries"""
        try:
            count = self.collection.count()  # type: ignore

            # Get some sample metadata for statistics
            if count > 0:
                sample_results = self.collection.get(  # type: ignore
                    include=["metadatas"], limit=min(100, count)
                )

                metadatas = sample_results["metadatas"]

                # Calculate statistics
                avg_code_length = (
                    sum(m.get("code_length", 0) for m in metadatas) / len(metadatas)  # type: ignore
                    if metadatas
                    else 0
                )
                avg_success_rate = (
                    sum(m.get("validation_success_rate", 0) for m in metadatas)  # type: ignore
                    / len(metadatas)
                    if metadatas
                    else 0
                )
                with_feedback_count = sum(
                    1 for m in metadatas if m.get("has_human_feedback", False)  # type: ignore
                )

                return {
                    "total_entries": count,
                    "average_code_length": int(avg_code_length),
                    "average_success_rate": round(avg_success_rate, 3),
                    "entries_with_human_feedback": with_feedback_count,
                    "feedback_percentage": (
                        round(with_feedback_count / len(metadatas) * 100, 1)
                        if metadatas
                        else 0
                    ),
                }
            else:
                return {
                    "total_entries": 0,
                    "average_code_length": 0,
                    "average_success_rate": 0,
                    "entries_with_human_feedback": 0,
                    "feedback_percentage": 0,
                }

        except Exception as e:
            print(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}

    def cleanup(self):
        """Cleanup resources"""
        if self.client:
            print("🧹 Cleaning up vector database resources")
            # ChromaDB doesn't require explicit cleanup, but you can add it here if needed
