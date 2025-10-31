"""
SysML v2 RAG Pipeline - Improved GitHub & Local Document Ingestion
Uses GitIngest Python package (proper method for AI agents)
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime
import re

# GitIngest - PROPER METHOD FOR AI AGENTS
from gitingest import ingest, ingest_async
import asyncio

# Vector store and embeddings
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


@dataclass
class SysMLDocument:
    """Represents a SysML document or code snippet"""

    content: str
    source: str  # github url or local path
    doc_type: str  # 'code', 'error_doc', 'example'
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    embedding: Optional[List[float]] = None


class GitIngestClient:
    """
    Client for GitIngest using the Python package (recommended for AI agents)
    CHANGED: Replaced HTTP API calls with gitingest package
    """

    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize GitIngest client

        Args:
            github_token: GitHub personal access token for private repos
        """
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")

    def ingest_repo(
        self,
        repo_url: str,
        file_patterns: Optional[List[str]] = None,
        max_file_size: int = 512000,  # 500KB default (SysML files can be large)
    ) -> Dict[str, str]:
        """
        Ingest SysML and KerML files from a GitHub repository

        Args:
            repo_url: GitHub repository URL
            file_patterns: List of file patterns to match (default: ['**/*.sysml', '**/*.kerml'])
            max_file_size: Maximum file size in bytes to process

        Returns:
            Dictionary mapping file paths to contents
        """

        try:
            # Use GitIngest Python package - the proper method!
            summary, tree, content = ingest(
                repo_url,
                include_patterns=file_patterns
                or ["**/*.sysml", "**/*.kerml", "*.sysml", "*.kerml"],  # type: ignore
                exclude_patterns=[
                    "*.lock",
                    "node_modules/*",
                    ".git/*",
                    "dist/*",
                    "build/*",
                    "*.log",
                    "*.tmp",
                    "*.class",
                    "*.jar",
                    "*.war",
                ],  # type: ignore
                max_file_size=max_file_size,
                token=self.github_token,
            )

            # Parse the structured content from GitIngest
            files = self._parse_gitingest_content(content, repo_url)

            if not files:
                print(f"  ⚠ No .sysml or .kerml files found in {repo_url}")
                print(
                    f"  💡 Tip: Check if the repository uses a different file structure"
                )
                return files

            # Log what was found
            sysml_count = sum(1 for f in files.keys() if f.endswith(".sysml"))
            kerml_count = sum(1 for f in files.keys() if f.endswith(".kerml"))

            print(f"  ✓ Found {sysml_count} .sysml and {kerml_count} .kerml files")
            print(f"  ✓ Total: {len(files)} files from {repo_url}")

            # Show first few file paths as examples
            if files:
                example_files = list(files.keys())[:3]
                print(f"  📁 Examples: {', '.join(example_files)}")

            return files

        except Exception as e:
            print(f"  ✗ Error ingesting {repo_url}: {e}")
            return {}

    async def ingest_repo_async(
        self,
        repo_url: str,
        file_patterns: Optional[List[str]] = None,
        max_file_size: int = 512000,  # 500KB default (SysML files can be large)
    ) -> Dict[str, str]:
        """Async version for batch processing multiple repos - searches recursively for .sysml and .kerml files"""
        # Always use SysML/KerML patterns with recursive search if not specified
        if file_patterns is None:
            file_patterns = [
                "**/*.sysml",  # Recursive search for .sysml files
                "**/*.kerml",  # Recursive search for .kerml files
                "*.sysml",  # Also check root directory
                "*.kerml",  # Also check root directory
            ]

        try:
            summary, tree, content = await ingest_async(
                repo_url,
                include_patterns=file_patterns
                or ["**/*.sysml", "**/*.kerml", "*.sysml", "*.kerml"],  # type: ignore
                exclude_patterns=[
                    "*.lock",
                    "node_modules/*",
                    ".git/*",
                    "dist/*",
                    "build/*",
                    "*.log",
                    "*.tmp",
                    "*.class",
                    "*.jar",
                    "*.war",
                ],  # type: ignore
                max_file_size=max_file_size,
                token=self.github_token,
            )

            files = self._parse_gitingest_content(content, repo_url)

            # Log what was found
            sysml_count = sum(1 for f in files.keys() if f.endswith(".sysml"))
            kerml_count = sum(1 for f in files.keys() if f.endswith(".kerml"))
            print(
                f"  ✓ Async found {sysml_count} .sysml and {kerml_count} .kerml files from {repo_url}"
            )

            return files

        except Exception as e:
            print(f"  ✗ Async error ingesting {repo_url}: {e}")
            return {}

    def _parse_gitingest_content(self, content: str, repo_url: str) -> Dict[str, str]:
        """
        Parse GitIngest's structured text output into a dict {filepath: content}.
        """
        files = {}
        sections = content.split("================================================")
        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            if header.startswith("FILE:"):
                filename = header.replace("FILE:", "").strip()
                file_body = sections[i + 1].strip() if i + 1 < len(sections) else ""
                files[filename] = file_body
        return files

    async def ingest_multiple_repos(
        self, repo_urls: List[str], file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Ingest multiple repositories concurrently using async

        Returns:
            Dictionary mapping repo URL to file dictionary
        """
        tasks = [self.ingest_repo_async(url, file_patterns) for url in repo_urls]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        repo_files = {}
        for repo_url, result in zip(repo_urls, results):
            if isinstance(result, Exception):
                print(f"  ✗ Failed to ingest {repo_url}: {result}")
                repo_files[repo_url] = {}
            else:
                repo_files[repo_url] = result

        return repo_files


class SysMLChunker:
    """Intelligent chunking for SysML v2 code"""

    @staticmethod
    def chunk_sysml_code(
        content: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[str]:
        """
        Chunk SysML code intelligently at definition boundaries

        Prioritizes splitting at:
        1. Package boundaries
        2. Part/action/state definitions
        3. Constraint/requirement definitions
        """
        chunks = []

        # Define splitting patterns (in order of priority)
        patterns = [
            r"\n(package\s+\w+\s*{)",  # Package definitions
            r"\n(part\s+def\s+\w+\s*{)",  # Part definitions
            r"\n(action\s+def\s+\w+\s*{)",  # Action definitions
            r"\n(state\s+def\s+\w+\s*{)",  # State definitions
            r"\n(requirement\s+def\s+\w+\s*{)",  # Requirement definitions
            r"\n(constraint\s+def\s+\w+\s*{)",  # Constraint definitions
            r"\n(interface\s+def\s+\w+\s*{)",  # Interface definitions
            r"\n(item\s+def\s+\w+\s*{)",  # Item definitions
            r"\n(attribute\s+def\s+\w+\s*{)",  # Attribute definitions
        ]

        # Find all split points
        split_points = [0]
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                split_points.append(match.start())

        split_points = sorted(set(split_points))
        split_points.append(len(content))

        # Create chunks with overlap
        for i in range(len(split_points) - 1):
            start = max(0, split_points[i] - overlap if i > 0 else 0)
            end = split_points[i + 1]
            chunk = content[start:end].strip()

            if len(chunk) > 50:  # Minimum chunk size
                chunks.append(chunk)

        # Fallback: if no smart splits found, do simple chunking
        if not chunks:
            chunks = SysMLChunker._simple_chunk(content, chunk_size, overlap)

        return chunks

    @staticmethod
    def _simple_chunk(content: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple fallback chunking by character count"""
        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    @staticmethod
    def chunk_error_docs(content: str) -> List[Dict[str, str]]:
        """
        Chunk error documentation by error sections
        Returns list of dicts with error code, category, and content
        """
        chunks = []

        # Split by error sections (### Error: ...)
        error_pattern = r"###\s+Error:\s+(.+?)(?=###\s+Error:|$)"
        matches = re.finditer(error_pattern, content, re.DOTALL)

        for match in matches:
            error_content = match.group(0).strip()

            # Extract metadata
            error_code_match = re.search(r"\*\*Error Code:\*\*\s+(\w+)", error_content)
            category_match = re.search(r"\*\*Category:\*\*\s+(.+)", error_content)
            severity_match = re.search(r"\*\*Severity:\*\*\s+(\w+)", error_content)

            chunk_metadata = {
                "error_code": error_code_match.group(1) if error_code_match else None,
                "category": category_match.group(1) if category_match else None,
                "severity": severity_match.group(1) if severity_match else None,
                "content": error_content,
            }
            chunks.append(chunk_metadata)

        return chunks


class SysMLRAGPipeline:
    """Main RAG pipeline for SysML v2"""

    def __init__(
        self,
        collection_name: str = "sysml_v2_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        github_token: Optional[str] = None,
    ):

        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "SysML v2 code and documentation"},
        )

        # Initialize GitIngest client with proper Python package
        self.git_client = GitIngestClient(github_token=github_token)
        self.chunker = SysMLChunker()

    def generate_chunk_id(self, content: str, source: str) -> str:
        """Generate unique ID for chunk"""
        combined = f"{source}:{content}"
        return hashlib.md5(combined.encode()).hexdigest()

    def ingest_github_repos(self, repo_urls: List[str], use_async: bool = True):
        """
        Ingest multiple GitHub repositories

        Args:
            repo_urls: List of GitHub repository URLs
            use_async: Use async processing for faster ingestion (default: True)
        """
        print(f"\nIngesting {len(repo_urls)} GitHub repositories...")

        all_documents = []

        if use_async and len(repo_urls) > 1:
            # Use async for multiple repos
            all_repo_files = asyncio.run(
                self.git_client.ingest_multiple_repos(repo_urls)
            )

            for repo_url, files in all_repo_files.items():
                all_documents.extend(self._process_repo_files(repo_url, files))
        else:
            # Sequential processing
            for repo_url in repo_urls:
                print(f"\n  Processing: {repo_url}")
                files = self.git_client.ingest_repo(repo_url)
                all_documents.extend(self._process_repo_files(repo_url, files))

        self._add_documents_to_vectorstore(all_documents)
        print(f"\n✓ Total: Ingested {len(all_documents)} code chunks from GitHub\n")

    def _process_repo_files(
        self, repo_url: str, files: Dict[str, str]
    ) -> List[SysMLDocument]:
        """Process files from a repository into documents"""
        documents = []

        for file_path, content in files.items():
            # Chunk the code intelligently
            chunks = self.chunker.chunk_sysml_code(content)

            for idx, chunk in enumerate(chunks):
                doc = SysMLDocument(
                    content=chunk,
                    source=f"{repo_url}/{file_path}",
                    doc_type="code",
                    metadata={
                        "repo": repo_url,
                        "file_path": file_path,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "file_type": self._get_file_type(file_path),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                documents.append(doc)

        return documents

    def _get_file_type(self, file_path: str) -> str:
        """Extract file type from path - specifically handles SysML and KerML"""
        ext = Path(file_path).suffix.lower()

        # Map extensions to descriptive types
        type_map = {
            ".sysml": "sysml",
            ".kerml": "kerml",
            ".md": "markdown",
            ".txt": "text",
        }

        return type_map.get(ext, ext.lstrip(".") if ext else "unknown")

    def ingest_local_documents(self, doc_paths: List[str]):
        """Ingest local documentation files"""
        print(f"\nIngesting {len(doc_paths)} local documents...")

        all_documents = []

        for doc_path in doc_paths:
            path = Path(doc_path)
            if not path.exists():
                print(f"  ⚠ Warning: {doc_path} not found")
                continue

            print(f"  Processing: {doc_path}")
            content = path.read_text(encoding="utf-8")

            # Check if it's error documentation
            if "Error:" in content and "**Error Code:**" in content:
                # Chunk by error sections
                error_chunks = self.chunker.chunk_error_docs(content)

                for error_chunk in error_chunks:
                    doc = SysMLDocument(
                        content=error_chunk["content"],
                        source=doc_path,
                        doc_type="error_doc",
                        metadata={
                            "error_code": error_chunk["error_code"],
                            "category": error_chunk["category"],
                            "severity": error_chunk["severity"],
                            "file_path": doc_path,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    all_documents.append(doc)
            else:
                # Regular documentation - chunk by size
                chunks = self.chunker.chunk_sysml_code(content, chunk_size=800)

                for idx, chunk in enumerate(chunks):
                    doc = SysMLDocument(
                        content=chunk,
                        source=doc_path,
                        doc_type="documentation",
                        metadata={
                            "file_path": doc_path,
                            "chunk_index": idx,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    all_documents.append(doc)

        self._add_documents_to_vectorstore(all_documents)
        print(f"✓ Ingested {len(all_documents)} documentation chunks\n")

    def _add_documents_to_vectorstore(self, documents: List[SysMLDocument]):
        """Add documents to ChromaDB vector store"""
        if not documents:
            return

        # Generate embeddings
        contents = [doc.content for doc in documents]
        print(f"  Generating embeddings for {len(contents)} chunks...")
        embeddings = self.embedding_model.encode(
            contents, show_progress_bar=True, batch_size=32
        )

        # Prepare for ChromaDB
        ids = []
        metadatas = []

        for doc, embedding in zip(documents, embeddings):
            chunk_id = self.generate_chunk_id(doc.content, doc.source)
            doc.chunk_id = chunk_id
            doc.embedding = embedding.tolist()

            ids.append(chunk_id)
            metadatas.append(
                {"source": doc.source, "doc_type": doc.doc_type, **doc.metadata}
            )

        # Add to collection
        print(f"  Storing in vector database...")
        self.collection.add(
            ids=ids,
            embeddings=[doc.embedding for doc in documents],  # type: ignore
            documents=contents,
            metadatas=metadatas,
        )

    def query(
        self, query_text: str, n_results: int = 5, filter_doc_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Query the RAG pipeline

        Args:
            query_text: User query
            n_results: Number of results to return
            filter_doc_type: Filter by document type ('code', 'error_doc', 'documentation')

        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0]

        # Build filter if needed
        where_filter = None
        if filter_doc_type:
            where_filter = {"doc_type": filter_doc_type}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter,  # type: ignore
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "content": results["documents"][0][i],  # type: ignore
                    "metadata": results["metadatas"][0][i],  # type: ignore
                    "distance": (
                        results["distances"][0][i] if "distances" in results else None  # type: ignore
                    ),
                }
            )

        return formatted_results

    def get_context_for_prompt(
        self, query: str, max_tokens: int = 4000, prioritize_errors: bool = True
    ) -> str:
        """
        Get context string for LLM prompt

        Args:
            query: User query
            max_tokens: Maximum tokens for context (rough estimate: 1 token ≈ 4 chars)
            prioritize_errors: Put error documentation first in context

        Returns:
            Formatted context string
        """
        # Query different document types
        code_results = self.query(query, n_results=3, filter_doc_type="code")
        error_results = self.query(query, n_results=2, filter_doc_type="error_doc")
        doc_results = self.query(query, n_results=2, filter_doc_type="documentation")

        context_parts = []
        current_length = 0
        max_chars = max_tokens * 4

        # Add error documentation first (higher priority for debugging)
        if prioritize_errors and error_results:
            context_parts.append("=== RELEVANT ERROR PATTERNS ===\n")
            for result in error_results:
                content = result["content"]
                if current_length + len(content) > max_chars:
                    break
                context_parts.append(content + "\n\n")
                current_length += len(content)

        # Add general documentation
        if doc_results and current_length < max_chars:
            context_parts.append("=== RELEVANT DOCUMENTATION ===\n")
            for result in doc_results:
                content = result["content"]
                if current_length + len(content) > max_chars:
                    break
                context_parts.append(content + "\n\n")
                current_length += len(content)

        # Add code examples
        if code_results and current_length < max_chars:
            context_parts.append("=== RELEVANT CODE EXAMPLES ===\n")
            for result in code_results:
                content = result["content"]
                if current_length + len(content) > max_chars:
                    break
                source = result["metadata"].get("source", "unknown")
                context_parts.append(f"Source: {source}\n{content}\n\n")
                current_length += len(content)

        return "".join(context_parts)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents"""
        count = self.collection.count()

        # Get sample to analyze doc types
        if count > 0:
            sample = self.collection.get(limit=min(count, 1000))
            doc_types = {}
            for metadata in sample["metadatas"]:  # type: ignore
                doc_type = metadata.get("doc_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            return {
                "total_chunks": count,
                "doc_type_distribution": doc_types,
                "collection_name": self.collection.name,
            }

        return {"total_chunks": 0}


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    print("=" * 70)
    print("SysML v2 RAG Pipeline - Improved GitIngest Integration")
    print("=" * 70)

    pipeline = SysMLRAGPipeline(
        github_token=os.getenv("GITHUB_TOKEN")  # Optional for private repos
    )

    # List of GitHub repositories with SysML v2 examples
    github_repos = [
        "https://github.com/Systems-Modeling/SysML-v2-Release",
        "https://github.com/GfSE/SysML-v2-Models",
        # Add more repos
    ]

    # Local documentation files
    local_docs = [
        "./documents/Types_Errors.md",
        # "./examples.md",
        # Add more local files
    ]

    # Ingest data
    print("\nStarting RAG pipeline ingestion...\n")

    # Ingest GitHub repos (with async for speed)
    pipeline.ingest_github_repos(github_repos, use_async=True)

    # Ingest local documents
    pipeline.ingest_local_documents(local_docs)

    # Show statistics
    stats = pipeline.get_collection_stats()
    print("\n" + "=" * 70)
    print("Ingestion Complete - Statistics")
    print("=" * 70)
    print(f"Total chunks indexed: {stats['total_chunks']}")
    print(f"Document type distribution: {stats.get('doc_type_distribution', {})}")

    # Example query
    print("\n" + "=" * 70)
    print("Example Query")
    print("=" * 70)

    query = "How do I define a constraint with proper syntax?"

    print(f"\nQuery: {query}\n")
    print("Retrieving context...")

    context = pipeline.get_context_for_prompt(query)

    print("\nRetrieved Context Preview (first 500 chars):")
    print("-" * 70)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("-" * 70)

    # Use this context in your LLM prompt
    llm_prompt = f"""You are a SysML v2 modeling expert. Generate complete, concrete models with NO placeholders.

CONTEXT:
{context}

USER QUERY:
{query}

Generate a complete SysML v2 solution:
"""

    print("\n" + "=" * 70)
    print("✓ LLM Prompt Ready")
    print("=" * 70)
    print(f"\nPrompt length: {len(llm_prompt)} characters")
    print(f" Prompt : {llm_prompt}")
