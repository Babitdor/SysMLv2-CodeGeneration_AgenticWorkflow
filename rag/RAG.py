from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
import pickle
import json
from langchain_classic.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import re
import numpy as np
import os
import shutil


class SysMLRetriever:
    def __init__(
        self,
        file_paths,
        chunk_size=512,
        chunk_overlap=128,
        force_rebuild=False,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_dir="./agents/cache",
        alpha=0.7,  # Weight for semantic vs lexical search
    ) -> None:
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.force_rebuild = force_rebuild
        self.cache_dir = cache_dir
        self.alpha = alpha

        self.reranker = CrossEncoder(rerank_model)
        self.vectorstore = None
        self.bm25 = None
        self.docs = []

        # Cache paths
        os.makedirs(self.cache_dir, exist_ok=True)
        self.docs_cache_path = os.path.join(self.cache_dir, "processed_docs.pkl")
        self.faiss_index_path = os.path.join(self.cache_dir, "faiss_index")
        self.bm25_cache_path = os.path.join(self.cache_dir, "bm25_index.pkl")

        # Initialize system
        if self.force_rebuild:
            self._clear_cache()

        self._load_and_prepare()
        self._build_vectorstore()
        self._build_bm25()

    def _clear_cache(self):
        """Clear all cached files"""
        print("Clearing cache...")
        for path in [self.docs_cache_path, self.faiss_index_path, self.bm25_cache_path]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

    def _load_and_prepare(self):
        """Load documents or use cached version"""
        if not self.force_rebuild and os.path.exists(self.docs_cache_path):
            print("Loading cached documents...")
            with open(self.docs_cache_path, "rb") as f:
                self.docs = pickle.load(f)
            print(f"Loaded {len(self.docs)} cached document chunks")
        else:
            print("Processing documents from scratch...")
            self._process_documents()
            with open(self.docs_cache_path, "wb") as f:
                pickle.dump(self.docs, f)
            print(f"Cached {len(self.docs)} processed document chunks")

    def _process_documents(self):
        """Process documents with simple, effective chunking"""
        docs = []

        # Get file list
        if isinstance(self.file_paths, str) and os.path.isdir(self.file_paths):
            all_files = [
                os.path.join(self.file_paths, f)
                for f in os.listdir(self.file_paths)
                if f.lower().endswith((".pdf", ".md", ".csv", ".json"))
            ]
        else:
            all_files = (
                self.file_paths
                if isinstance(self.file_paths, list)
                else [self.file_paths]
            )

        if not all_files:
            raise ValueError(f"No supported files found: {self.file_paths}")

        # Load each file
        for path in all_files:
            try:
                if path.endswith(".pdf"):
                    loader = PDFPlumberLoader(path)
                    docs.extend(loader.load())
                elif path.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(path)
                    docs.extend(loader.load())
                elif path.endswith(".csv"):
                    loader = CSVLoader(file_path=path, encoding="utf-8")
                    docs.extend(loader.load())
                elif path.endswith(".json"):
                    json_docs = self._load_json_error_types(path)
                    docs.extend(json_docs)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue

        # Filter valid documents
        docs_list = [
            d for d in docs if isinstance(d, Document) and d.page_content.strip()
        ]
        if not docs_list:
            raise ValueError("No valid documents found after loading.")

        # Simple chunking
        self.docs = self._chunk_documents(docs_list)

    def _chunk_documents(self, documents):
        """Simple but effective document chunking"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = []
        for doc in documents:
            doc_chunks = splitter.split_text(doc.page_content)

            for i, chunk_text in enumerate(doc_chunks):
                if len(chunk_text.strip()) < 20:  # Skip tiny chunks
                    continue

                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks),
                    },
                )
                chunks.append(chunk)

        return chunks

    def _load_json_error_types(self, json_path):
        """Load and process JSON error types dataset"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file {json_path}: {e}")
            return []

        documents = []
        dataset = data.get("sysml_v2_errors_dataset", {})

        # Process error categories
        for category in dataset.get("error_categories", []):
            category_name = category.get("category_name", "Unknown")

            # Process individual errors
            for error in category.get("errors", []):
                error_doc = self._create_error_document(error, category_name, json_path)
                if error_doc:
                    documents.append(error_doc)

        return documents

    def _create_error_document(self, error, category_name, source_path):
        """Create a focused document for an individual error"""
        error_id = error.get("error_id", "unknown")
        error_name = error.get("error_name", "Unknown Error")
        problem = error.get("problem", "")
        wrong_example = error.get("wrong_example", "")
        correct_example = error.get("correct_example", "")
        explanation = error.get("explanation", "")

        # Build focused content
        content = f"Error {error_id}: {error_name}\n"
        content += f"Category: {category_name}\n\n"

        if problem:
            content += f"Problem: {problem}\n\n"

        if explanation:
            content += f"Solution: {explanation}\n\n"

        if correct_example:
            content += f"Correct syntax:\n{correct_example}\n\n"

        if wrong_example:
            content += f"Wrong syntax:\n{wrong_example}\n"

        return Document(
            page_content=content,
            metadata={
                "source": source_path,
                "error_id": error_id,
                "error_name": error_name,
                "category_name": category_name,
                "has_examples": bool(wrong_example and correct_example),
            },
        )

    def _build_vectorstore(self):
        """Build or load FAISS vectorstore"""
        if not self.force_rebuild and os.path.exists(self.faiss_index_path):
            print("Loading existing FAISS index...")
            try:
                self.vectorstore = FAISS.load_local(
                    self.faiss_index_path,
                    HuggingFaceEmbeddings(),
                    allow_dangerous_deserialization=True,
                )
                print(
                    f"Loaded FAISS index with {self.vectorstore.index.ntotal} vectors"
                )
            except Exception as e:
                print(f"Warning: Failed to load FAISS index: {e}")
                print("Building new index...")
                self._build_new_vectorstore()
        else:
            self._build_new_vectorstore()

    def _build_new_vectorstore(self):
        """Build new FAISS vectorstore"""
        print("Building new FAISS index...")
        self.vectorstore = FAISS.from_documents(
            documents=self.docs, embedding=HuggingFaceEmbeddings()
        )
        self.vectorstore.save_local(self.faiss_index_path)
        print(
            f"Built and saved FAISS index with {self.vectorstore.index.ntotal} vectors"
        )

    def _build_bm25(self):
        """Build or load BM25 index"""
        if not self.force_rebuild and os.path.exists(self.bm25_cache_path):
            print("Loading existing BM25 index...")
            try:
                with open(self.bm25_cache_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                print(f"Loaded BM25 index for {len(self.docs)} documents")
            except Exception as e:
                print(f"Warning: Failed to load BM25 index: {e}")
                print("Building new BM25 index...")
                self._build_new_bm25()
        else:
            self._build_new_bm25()

    def _build_new_bm25(self):
        """Build new BM25 index"""
        print("Building new BM25 index...")
        tokenized_corpus = [doc.page_content.split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

        with open(self.bm25_cache_path, "wb") as f:
            pickle.dump(self.bm25, f)
        print(f"Built and saved BM25 index for {len(self.docs)} documents")

    def _semantic_search(self, query, k=5):
        """Simple semantic search using FAISS"""
        return self.vectorstore.similarity_search(query, k=k)  # type: ignore

    def _hybrid_search(self, query, k=10):
        """Hybrid search combining semantic and lexical search"""
        try:
            # Get semantic results with scores
            semantic_results = self.vectorstore.similarity_search_with_score(  # type: ignore
                query, k=k * 2
            )
            semantic_dict = {doc.page_content: score for doc, score in semantic_results}

            # Get BM25 scores
            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens)  # type: ignore
            bm25_dict = {
                self.docs[i].page_content: bm25_scores[i] for i in range(len(self.docs))
            }

            # Normalize scores
            sem_scores = list(semantic_dict.values())
            bm_scores = list(bm25_dict.values())

            sem_max = max(sem_scores) if sem_scores else 1.0
            bm_max = max(bm_scores) if bm_scores else 1.0

            # Combine scores
            results = []
            for doc in self.docs:
                sem_score = semantic_dict.get(doc.page_content, 0.0) / sem_max
                bm_score = bm25_dict.get(doc.page_content, 0.0) / bm_max

                final_score = self.alpha * sem_score + (1 - self.alpha) * bm_score
                results.append((doc, final_score))

            # Sort and return top k
            ranked_results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
            return [doc for doc, _ in ranked_results]

        except Exception as e:
            print(f"Warning: Hybrid search failed: {e}")
            # Fallback to semantic search
            return self._semantic_search(query, k)

    def _rerank_search(self, query, k=5):
        """Hybrid search with cross-encoder reranking"""
        candidates = self._hybrid_search(query, k * 3)
        if not candidates:
            return []

        try:
            # Rerank with cross-encoder
            pairs = [[query, doc.page_content] for doc in candidates]
            scores = self.reranker.predict(pairs)

            # Sort by reranking scores
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[
                :k
            ]

            # Add ranking information to results
            reranked_docs = []
            for i, (doc, score) in enumerate(ranked, start=1):
                # Clean way to add ranking info
                enhanced_doc = Document(
                    page_content=f"[Rank {i} | Score: {score:.4f}]\n{doc.page_content}",
                    metadata=doc.metadata,
                )
                reranked_docs.append(enhanced_doc)

            return reranked_docs

        except Exception as e:
            print(f"Warning: Reranking failed: {e}")
            return candidates[:k]

    def query(self, text, k=5, mode="hybrid"):
        """
        Main query method with three modes:
        - "fast": Semantic search only (fastest)
        - "hybrid": BM25 + semantic search (balanced)
        - "rerank": Hybrid + cross-encoder reranking (best quality)
        """
        if mode == "fast":
            return self._semantic_search(text, k)
        elif mode == "hybrid":
            return self._hybrid_search(text, k)
        elif mode == "rerank":
            return self._rerank_search(text, k)
        else:
            raise ValueError("Mode must be 'fast', 'hybrid', or 'rerank'")

    def query_by_error_id(self, error_id):
        """Find specific error by ID"""
        matches = [doc for doc in self.docs if doc.metadata.get("error_id") == error_id]
        return matches

    def query_by_category(self, category_name, k=10):
        """Find errors by category"""
        matches = [
            doc
            for doc in self.docs
            if category_name.lower() in doc.metadata.get("category_name", "").lower()
        ]
        return matches[:k]

    def get_statistics(self):
        """Get basic statistics about the loaded documents"""
        total_docs = len(self.docs)
        error_docs = len([d for d in self.docs if d.metadata.get("error_id")])
        categories = set(
            d.metadata.get("category_name")
            for d in self.docs
            if d.metadata.get("category_name")
        )
        avg_size = (
            sum(len(d.page_content) for d in self.docs) / total_docs
            if total_docs > 0
            else 0
        )

        return {
            "total_documents": total_docs,
            "error_documents": error_docs,
            "categories": list(categories),
            "average_chunk_size": avg_size,
        }


# def test_streamlined_retriever():
#     """Test the streamlined retriever"""
#     print("Testing Streamlined SysML Retriever")
#     print("=" * 40)

#     try:
#         # Initialize retriever
#         retriever = SysMLRetriever(
#             file_paths="./agents/rag_data",  # Adjust path as needed
#             chunk_size=512,
#             chunk_overlap=128,
#             force_rebuild=False,
#         )

#         # Get statistics
#         stats = retriever.get_statistics()
#         print(f"Loaded {stats['total_documents']} documents")
#         print(f"Error documents: {stats['error_documents']}")
#         print(f"Categories: {len(stats['categories'])}")
#         print(f"Average chunk size: {stats['average_chunk_size']:.0f} chars")

#         # Test different search modes
#         test_query = "part definition syntax error"

#         print(f"\nTesting query: '{test_query}'")
#         print("-" * 30)

#         for mode in ["fast", "hybrid", "rerank"]:
#             results = retriever.query(test_query, k=3, mode=mode)
#             print(f"{mode.upper()} mode: {len(results)} results")

#             if results:
#                 first_result = results[0].page_content
#                 # Clean ranking prefix if present
#                 if first_result.startswith("[Rank"):
#                     first_result = (
#                         first_result.split("\n", 1)[1]
#                         if "\n" in first_result
#                         else first_result
#                     )
#                 print(f"  Top result: {first_result[:100]}...")

#         print("\nTest completed successfully!")

#     except Exception as e:
#         print(f"Test failed: {e}")
#         print("This might be due to missing data files.")


# if __name__ == "__main__":
#     test_streamlined_retriever()
