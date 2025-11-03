import os
from typing import List, Optional, Dict
from gitingest import ingest


def parse_gitingest_content(content: str, repo_url: str) -> Dict[str, str]:
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


def ingest_repo(
    repo_url: str,
    file_patterns: Optional[List[str]] = None,
    max_file_size: int = 512000,  # 500KB default
    github_token: Optional[str] = None,
) -> Dict[str, str]:
    """
    Ingest SysML and KerML files from a GitHub repository.
    """
    try:
        # Use GitIngest Python package
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
            token=github_token,
        )

        # Parse structured content
        files = parse_gitingest_content(content, repo_url)

        if not files:
            print(f"⚠ No .sysml or .kerml files found in {repo_url}")
            return {}

        sysml_count = sum(1 for f in files if f.endswith(".sysml"))
        kerml_count = sum(1 for f in files if f.endswith(".kerml"))

        print(f"✓ Found {sysml_count} .sysml and {kerml_count} .kerml files")
        print(f"✓ Total: {len(files)} files from {repo_url}")

        print("📁 Example files:")
        for i, fname in enumerate(list(files.keys())[:3]):
            print(f"   {i+1}. {fname}")

        return files

    except Exception as e:
        print(f"✗ Error ingesting {repo_url}: {e}")
        return {}


if __name__ == "__main__":
    # --- Configuration ---
    repo_url = input("Enter GitHub repo URL: ").strip()
    github_token = os.getenv("GITHUB_TOKEN")  # optional, for private repos

    print(f"\n🚀 Ingesting repository: {repo_url}\n")
    files = ingest_repo(repo_url, github_token=github_token)

    if files:
        print("\n✅ Ingestion complete. Example content snippet:")
        first_file = next(iter(files))
        print(f"\n--- {first_file} ---\n{files[first_file][:400]}...\n")
    else:
        print("\n❌ No SysML/KerML files ingested.")
