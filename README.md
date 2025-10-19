# Enhanced SysML Workflow Assistant

A sophisticated AI-powered system for generating and validating SysML v2 (Systems Modeling Language) code using multi-agent architecture with Retrieval Augmented Generation (RAG) and vector database integration.

## 🎯 Overview

The Enhanced SysML Workflow Assistant automates the creation, validation, and refinement of SysML v2 models through an intelligent pipeline of specialized agents. It combines language models, semantic search, and human feedback to produce high-quality systems modeling code.

### Key Capabilities

- **Intelligent Code Generation**: AI-powered SysML v2 code generation with context awareness
- **Automated Validation**: Multi-step validation with syntax and semantic checks
- **Error Correction**: Automatic detection and fixing of code issues
- **Knowledge Management**: RAG system for retrieving relevant patterns and examples
- **Solution Memory**: Vector database for storing and learning from approved solutions
- **Human Oversight**: Optional human approval with feedback integration
- **Analytics Dashboard**: Real-time performance tracking and insights

## 🏗️ Architecture

### Agent Components

The workflow uses a multi-agent system orchestrated by LangGraph:

- **QueryAgent**: Processes and structures natural language requirements
- **SysMLAgent**: Generates SysML v2 code with knowledge base context
- **ValidatorAgent**: Validates code syntax and semantics using Jupyter kernel
- **CodeCorrectionAgent**: Analyzes errors and provides fixes
- **HumanApprovalAgent**: Manages human review and feedback
- **KnowledgeBase**: Vector database for approved solutions
- **RAG System**: Retrieves relevant patterns and error solutions

### Workflow Pipeline

```
Query → Processing → RAG Enhancement → Code Generation → Validation
                                              ↓
                                         Error Correction ← (if needed)
                                              ↓
                                       Human Approval
                                              ↓
                                      Knowledge Update
```

## 📋 Requirements

### System Dependencies

- Python 3.9+
- Jupyter with SysML kernel
- Ollama (for local LLM inference)
- ChromaDB (vector database)
- Java 11+ (for SysML kernel)

### Python Dependencies

See `requirements.txt` for complete list. Key packages:

```
langchain-ollama
langchain-huggingface
langchain-community
langgraph
chromadb
sentence-transformers
streamlit
jupyter-client
torch
transformers
```

## 🚀 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd enhanced-sysml-workflow
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install SysML Kernel (Optional but Recommended)

```bash
conda install conda-forge::jupyter-sysml-kernel
```

### 5. Install Ollama

Download from [ollama.ai](https://ollama.ai) and pull the SysML model:

```bash
ollama pull SysML-V2-llama3.1:latest
```

## 💻 Usage

### Quick Start - Streamlit UI

```bash
streamlit run Run.py
```

Then:

1. Configure settings in the sidebar
2. Click "Initialize Workflow"
3. Enter your SysML modeling request
4. Review results and download generated code

### Programmatic Usage

```python
from AgentWorkflow import EnhancedSysMLWorkflow

# Initialize workflow
workflow = EnhancedSysMLWorkflow(
    enable_rag=True,
    enable_vector_storage=True,
    model_name="SysML-V2-llama3.1:latest",
    temperature=0.15
)

# Run workflow
result = workflow.run(
    query="Create a SysML model for an automotive brake system with ABS",
    max_iterations=5
)

# Access results
print(f"Success: {result['success']}")
print(f"Generated Code:\n{result['final_code']}")
print(f"Approval Status: {result['approval_status']}")

# Search similar solutions
similar = workflow.search_similar_solutions("automotive system", n_results=5)

# Cleanup
workflow.cleanup()
```

### Command Line Testing

```bash
# Test individual components
python AgentWorkflow.py        # Test enhanced workflow
python SysMLAgent.py           # Test code generation
python ValidatorAgent.py       # Test validation
python CodeCorrectionAgent.py  # Test error correction
```

## 📁 Project Structure

```
.
├── AgentWorkflow.py          # Main workflow orchestration
├── SysMLAgent.py             # Code generation agent
├── ValidatorAgent.py         # Validation agent
├── CodeCorrectionAgent.py    # Error correction agent
├── QueryAgent.py             # Query processing agent
├── HumanApprovalAgent.py     # Human approval handler
├── KnowledgeBase.py          # Vector database manager
├── RAG.py                    # RAG system implementation
├── States.py                 # State management & data models
├── Run.py                    # Streamlit application
├── prompt.yaml               # Agent prompts and configurations
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### prompt.yaml

Customize agent prompts and system instructions:

```yaml
SysML-Assistant: |
  [System prompt for code generation]

Query-Agent: |
  [System prompt for query processing]

Code-Correction-Agent: |
  [System prompt for error correction]
```

### Environment Variables

```bash
ANONYMIZED_TELEMETRY=False    # Disable ChromaDB telemetry
PYTHONWARNINGS=ignore         # Suppress warnings
JAVA_TOOL_OPTIONS=...         # Java configuration for SysML kernel
```

## 📊 Features

### RAG System

Retrieves relevant SysML patterns and error solutions using:

- **Semantic Search**: Vector similarity with FAISS
- **Lexical Search**: BM25 token-based retrieval
- **Hybrid Search**: Combined semantic + lexical
- **Reranking**: Cross-encoder model for result ranking

### Vector Database

Stores approved solutions with metadata:

- Task description and requirements
- Generated SysML code
- Validation results and success rate
- Human feedback and iterations
- Embedding for semantic search

### Validation

Multi-layer validation approach:

- Syntax validation via SysML Jupyter kernel
- Semantic analysis of model structure
- Error detection and categorization
- Detailed error messages with line numbers

## 🎓 Examples

### Example 1: Simple Vehicle System

```python
query = "Create a SysML model for a simple vehicle with engine and transmission"
result = workflow.run(query)
print(result['final_code'])
```

### Example 2: With Error Recovery

The system automatically detects validation errors and attempts fixes:

```python
# Even if initial code has issues, the system will:
# 1. Detect errors via validator
# 2. Provide correction guidance
# 3. Regenerate corrected code
# 4. Re-validate automatically
result = workflow.run(query, max_iterations=5)
```

### Example 3: Finding Similar Solutions

```python
similar = workflow.search_similar_solutions(
    "automotive system with sensors",
    n_results=3
)

for solution in similar:
    print(f"Task: {solution['task']}")
    print(f"Similarity: {solution['similarity_score']:.3f}")
    print(f"Code:\n{solution['full_entry']['generated_code']}")
```

## 📈 Performance Tips

1. **Enable RAG**: Use retrieval augmented generation for better context
2. **Vector Database**: Enable storage to learn from previous solutions
3. **Lower Temperature**: Set temperature to 0.15 for more deterministic output
4. **Clear Queries**: Be specific about requirements and constraints
5. **Monitor Analytics**: Use the dashboard to identify patterns and optimize

## 🐛 Troubleshooting

### SysML Kernel Not Found

```bash
# Install the kernel
conda install conda-forge::jupyter-sysml-kernel

# Verify installation
jupyter kernelspec list
```

### Ollama Model Not Available

```bash
# List available models
ollama list

# Pull the model
ollama pull SysML-V2-llama3.1:latest

# Start Ollama service
ollama serve
```

### Vector Database Issues

```python
# Force rebuild cache
workflow = EnhancedSysMLWorkflow(force_rebuild_rag=True)

# Clear database
import shutil
shutil.rmtree("./sysml_knowledge_base")
```

### Memory Issues

- Reduce chunk size in RAG configuration
- Limit vector database results
- Enable garbage collection in code
- Use batch processing for multiple queries

## 🔮 Future Enhancements

- Multi-model support (GPT-4, Claude, etc.)
- Web API interface
- Collaborative features with version control
- Advanced constraint optimization
- Integration with SysML modeling tools
- Real-time collaboration support
- Custom agent development framework

## 📚 References

- [SysML v2 Specification](https://www.omg.org/sysml/)
- [LangChain Documentation](https://python.langchain.com)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ChromaDB Documentation](https://docs.trychroma.com)

---

**Last Updated**: 2025  
**Version**: 1.0.0  
**Status**: Active Development
