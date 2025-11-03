import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_graph import SysMLWorkflow


def test_enhanced_workflow():
    """Test the enhanced workflow"""
    print("=" * 80)
    print("SIMPLIFIED SYSML WORKFLOW TEST")
    print("=" * 80)

    workflow = None
    try:
        print("\n1. Initializing Simplified SysML Workflow...")

        workflow = SysMLWorkflow(
            enable_rag=True,
            query_agent_llm={
                "provider": "ollama",
                "model": "qwen3:1.7b",
                "temperature": 0.15,
                "num_ctx": 2048,
            },
            template_agent_llm={
                "provider": "ollama",
                "model": "QwenCoder2.5-7B-SysML:latest",
                "temperature": 0.15,
                "num_ctx": 16000,
            },
            sysml_agent_llm={
                "provider": "ollama",
                "model": "qwen3-coder:480b-cloud",
                "temperature": 0.15,
                "num_ctx": 16000,
            },
            agents_prompt="agents/prompt/prompt.yaml",
            rag_persist_directory="./rag/chroma_db",
            rag_collection_name="sysml_v2_knowledge",
            enable_approved_solutions_db=True,
            approved_solutions_persist_dir="./rag/approved_solutions",
            context_length=16000,
        )
        print("✅ Simplified workflow initialized successfully")

        # Display system status
        print("\n2. System Status:")
        print(
            f"   RAG System: {'✅ Enabled' if workflow.enable_rag else '❌ Disabled'}"
        )
        print(
            f"   Approved Solutions DB: {'✅ Enabled' if workflow.enable_approved_solutions else '❌ Disabled'}"
        )

        if workflow.enable_rag:
            rag_stats = workflow.get_rag_statistics()
            print(f"\n   RAG Pipeline Statistics:")
            print(f"   - Total chunks: {rag_stats.get('total_chunks', 0)}")

        if workflow.enable_approved_solutions:
            approved_stats = workflow.get_approved_db_statistics()
            print(f"\n   Approved Solutions Database:")
            print(f"   - Total entries: {approved_stats.get('total_entries', 0)}")
            if approved_stats.get("sample_ids"):
                print(f"   - Sample IDs: {approved_stats['sample_ids']}")

        # Test query
        test_query = "Create a SysML package for a Vehicle"
        print(f"\n3. Running workflow with test query:")
        print(f"   '{test_query}'")

        result = workflow.run(test_query, max_iterations=5)

        if result.get("success"):
            print("\n4. Workflow Results:")
            print(f"   Status: ✅ Success")
            print(f"   Iterations: {result.get('iteration', 0)}")
            print(f"   Approval Status: {result.get('approval_status', 'unknown')}")

            if result.get("code"):
                print(f"   Final Code Length: {len(result['code'])} characters")
                print("\n5. Generated Code Preview:")
                print("-" * 50)
                code_lines = result["code"].split("\n")[:15]
                for line in code_lines:
                    print(f"   {line}")
                if len(result["code"].split("\n")) > 15:
                    print("   ... (truncated)")
                print("-" * 50)
        else:
            print(f"\n4. ❌ Workflow failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("\n6. Cleanup:")
        if workflow:
            workflow.cleanup()
        print("   ✅ Resources cleaned up successfully")


if __name__ == "__main__":
    test_enhanced_workflow()
