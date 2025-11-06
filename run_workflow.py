import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import traceback

# Import your workflow class
try:
    from build_graph import SysMLWorkflow

    WORKFLOW_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import SysMLWorkflow: {e}")
    WORKFLOW_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SysML AI Workflow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .code-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


class StreamlitSysMLApp:
    """Streamlit app wrapper for Enhanced SysML Workflow"""

    def __init__(self):
        self.workflow = st.session_state.get("workflow", None)
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "workflow_initialized" not in st.session_state:
            st.session_state.workflow_initialized = False
        if "workflow_results" not in st.session_state:
            st.session_state.workflow_results = []
        if "current_result" not in st.session_state:
            st.session_state.current_result = None
        if "system_stats" not in st.session_state:
            st.session_state.system_stats = {}
        if "similar_solutions" not in st.session_state:
            st.session_state.similar_solutions = []

    def render_header(self):
        """Render the main header"""
        st.markdown(
            '<h1 class="main-header">🤖 SysML AI Workflow Assistant</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "**Flow:** Query Agent → Template Agent → SysML Agent → Human Approval"
        )
        st.markdown("---")

    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.header("⚙️ Configuration")

        # Workflow settings
        st.sidebar.subheader("Workflow Settings")

        model_name = st.sidebar.text_input(
            "SysML Agent Model",
            value="qwen3-coder:480b-cloud",
            help="Model for SysML Agent (refinement & validation)",
        )

        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.15,
            step=0.05,
            help="Controls randomness in generation",
        )

        context_length = st.sidebar.number_input(
            "Context Length",
            min_value=4096,
            max_value=128000,
            value=16000,
            step=1024,
            help="Maximum context window size",
        )

        max_iterations = st.sidebar.slider(
            "Max Iterations",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum retry iterations",
        )

        # Advanced features
        st.sidebar.subheader("Advanced Features")

        enable_rag = st.sidebar.checkbox(
            "Enable RAG System",
            value=True,
            help="Use RAG for context enhancement (tool in SysMLAgent)",
        )

        enable_approved_db = st.sidebar.checkbox(
            "Enable Approved Solutions DB",
            value=True,
            help="Store and retrieve approved solutions",
        )

        # Data paths
        st.sidebar.subheader("Data Paths")

        config_path = st.sidebar.text_input(
            "Config Path",
            value="agents/prompt/prompt.yaml",
            help="Path to agent configuration file",
        )

        rag_persist_dir = st.sidebar.text_input(
            "RAG Database Path",
            value="./rag/chroma_db",
            help="Path to RAG vector database",
        )

        rag_collection_name = st.sidebar.text_input(
            "RAG Collection Name",
            value="sysml_v2_knowledge",
            help="Name of the RAG collection",
        )

        approved_persist_dir = st.sidebar.text_input(
            "Approved Solutions DB Path",
            value="./rag/approved_solutions",
            help="Path to approved solutions database",
        )

        # Initialize/Reset workflow
        st.sidebar.subheader("Workflow Control")

        if st.sidebar.button("🚀 Initialize Workflow", type="primary"):
            self.initialize_workflow(
                model_name=model_name,
                temperature=temperature,
                context_length=context_length,
                enable_rag=enable_rag,
                enable_approved_db=enable_approved_db,
                config_path=config_path,
                rag_persist_dir=rag_persist_dir,
                rag_collection_name=rag_collection_name,
                approved_persist_dir=approved_persist_dir,
            )

        if st.sidebar.button("🔄 Reset Workflow"):
            self.reset_workflow()

        # System status
        if st.session_state.workflow_initialized:
            st.sidebar.subheader("🟢 System Status")
            st.sidebar.success("Workflow Ready")

            # Show RAG database info
            if enable_rag:
                stats = st.session_state.system_stats.get("rag", {})
                if stats.get("total_chunks", 0) == 0:
                    st.sidebar.warning("⚠️ RAG database is empty")
                    st.sidebar.info("💡 Run: python Injestion.py")
                else:
                    st.sidebar.info(f"📚 {stats.get('total_chunks', 0)} RAG chunks")

            # Show approved solutions info
            if enable_approved_db:
                stats = st.session_state.system_stats.get("approved_db", {})
                st.sidebar.info(
                    f"💾 {stats.get('total_entries', 0)} approved solutions"
                )
        else:
            st.sidebar.subheader("🔴 System Status")
            st.sidebar.error("Workflow Not Initialized")

        return {
            "model_name": model_name,
            "temperature": temperature,
            "context_length": context_length,
            "max_iterations": max_iterations,
            "enable_rag": enable_rag,
            "enable_approved_db": enable_approved_db,
            "config_path": config_path,
            "rag_persist_dir": rag_persist_dir,
            "rag_collection_name": rag_collection_name,
            "approved_persist_dir": approved_persist_dir,
        }

    def initialize_workflow(self, **kwargs):
        """Initialize the workflow with given parameters"""
        if not WORKFLOW_AVAILABLE:
            st.error("SysML Workflow not available. Please check imports.")
            return

        try:
            with st.spinner("Initializing SysML Workflow..."):
                workflow = SysMLWorkflow(  # type: ignore
                    agents_prompt=kwargs["config_path"],
                    model_name=kwargs["model_name"],
                    temperature=kwargs["temperature"],
                    context_length=kwargs["context_length"],
                    enable_rag=kwargs["enable_rag"],
                    rag_persist_directory=kwargs["rag_persist_dir"],
                    rag_collection_name=kwargs["rag_collection_name"],
                    enable_approved_solutions_db=kwargs["enable_approved_db"],
                    approved_solutions_persist_dir=kwargs["approved_persist_dir"],
                )

                st.session_state.workflow = workflow
                st.session_state.workflow_initialized = True
                self.workflow = workflow

                self.update_system_stats()

                # Check RAG database status
                rag_stats = st.session_state.system_stats.get("rag", {})
                if kwargs["enable_rag"] and rag_stats.get("total_chunks", 0) == 0:
                    st.warning(
                        "⚠️ RAG database is empty. Run `python Injestion.py` to populate it."
                    )

                st.success("✅ Workflow initialized successfully!")
                st.info(
                    "Flow: Query Agent → Template Agent → SysML Agent (with RAG tool)"
                )

        except Exception as e:
            st.error(f"❌ Failed to initialize workflow: {str(e)}")
            st.exception(e)

    def reset_workflow(self):
        """Reset the workflow and clear session state"""
        if self.workflow:
            try:
                self.workflow.cleanup()
            except:
                pass

        self.workflow = None
        st.session_state.workflow = None
        st.session_state.workflow_initialized = False
        st.session_state.workflow_results = []
        st.session_state.current_result = None
        st.session_state.system_stats = {}
        st.session_state.similar_solutions = []
        st.success("🔄 Workflow reset successfully!")
        st.rerun()

    def update_system_stats(self):
        """Update system statistics"""
        if not self.workflow:
            return

        try:
            rag_stats = self.workflow.get_rag_statistics()
            approved_stats = self.workflow.get_approved_db_statistics()

            st.session_state.system_stats = {
                "rag": rag_stats,
                "approved_db": approved_stats,
                "total_results": len(st.session_state.workflow_results),
                "last_updated": datetime.now(),
            }
        except Exception as e:
            st.warning(f"Could not update system stats: {e}")

    def render_system_overview(self):
        """Render system overview and statistics"""
        st.subheader("📊 System Overview")

        if not st.session_state.workflow_initialized:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.info(
                    "👈 **Please initialize the workflow in the sidebar to begin.**"
                )
                st.markdown(
                    """
                **Quick Start Guide:**
                1. 📝 Review settings in the sidebar (left panel)
                2. 🚀 Click "Initialize Workflow" button
                3. ⏳ Wait for initialization to complete
                4. 🎯 Start using the system!
                
                **⚠️ Important:** Run `python Injestion.py` first to populate the RAG database!
                
                **New Architecture:**
                - Query Agent → Analyzes user intent
                - Template Agent → Creates SysML structure
                - SysML Agent → Refines & validates (uses RAG as tool)
                - Human Approval → Quality assurance
                """
                )

            with col2:
                st.metric("Workflow Status", "Not Initialized", "⚠️")
                st.metric("RAG System", "N/A")
                st.metric("Total Queries", "0")

            return

        # Create metrics columns (when initialized)
        col1, col2, col3, col4 = st.columns(4)

        stats = st.session_state.system_stats

        with col1:
            rag_enabled = stats.get("rag", {}).get("enabled", False)
            rag_chunks = stats.get("rag", {}).get("total_chunks", 0)

            if rag_enabled:
                st.metric(
                    label="RAG System",
                    value="Enabled",
                    delta=f"{rag_chunks} chunks",
                    help="Tool in SysMLAgent",
                )
            else:
                st.metric(label="RAG System", value="Disabled")

        with col2:
            approved_enabled = stats.get("approved_db", {}).get("enabled", False)
            approved_entries = stats.get("approved_db", {}).get("total_entries", 0)

            st.metric(
                "Approved Solutions",
                "Enabled" if approved_enabled else "Disabled",
                f"{approved_entries} entries" if approved_enabled else None,
            )

        with col3:
            total_results = stats.get("total_results", 0)
            successful_results = sum(
                1 for r in st.session_state.workflow_results if r.get("success", False)
            )
            success_rate = (
                (successful_results / total_results * 100) if total_results > 0 else 0
            )
            st.metric(
                "Total Queries", total_results, f"{success_rate:.1f}% success rate"
            )

        with col4:
            last_updated = stats.get("last_updated")
            if last_updated:
                time_str = last_updated.strftime("%H:%M:%S")
                st.metric("Last Updated", time_str, None)
            else:
                st.metric("System Status", "Online", "✅")

        # Show RAG database warning if empty
        if (
            stats.get("rag", {}).get("enabled")
            and stats.get("rag", {}).get("total_chunks", 0) == 0
        ):
            st.markdown(
                '<div class="warning-box">⚠️ <b>RAG Database is Empty!</b><br>'
                "Run <code>python Injestion.py</code> to populate it with SysML examples.</div>",
                unsafe_allow_html=True,
            )

    def render_query_interface(self):
        """Render the main query interface"""
        if not st.session_state.workflow_initialized:
            return

        st.subheader("💬 SysML Query Interface")

        # Query input
        query = st.text_area(
            "Enter your SysML modeling request:",
            placeholder="e.g., Create a SysML package for a vehicle with engine and transmission",
            height=100,
            help="Describe what SysML model you want to generate.",
        )

        # Execution options
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.button(
                "🚀 Generate SysML Code", type="primary", disabled=not query.strip()
            ):
                self.execute_workflow(
                    query.strip(), st.session_state.get("max_iter", 5)
                )

        with col2:
            max_iter = st.number_input(
                "Max Iterations", min_value=1, max_value=10, value=5, key="max_iter"
            )

        with col3:
            search_similar = st.button("🔍 Search Similar", disabled=not query.strip())

        if search_similar:
            self.search_similar_solutions(query.strip())

    def execute_workflow(self, query: str, max_iterations: int = 5):
        """Execute the SysML workflow"""
        if not self.workflow:
            st.error("Workflow not initialized!")
            return

        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔄 Executing workflow...")
            progress_bar.progress(20)

            # Execute workflow
            start_time = time.time()
            result = self.workflow.run(query, max_iterations=max_iterations)
            execution_time = time.time() - start_time

            progress_bar.progress(100)
            status_text.text("✅ Workflow completed!")

            # Add execution metadata
            result["execution_time"] = execution_time
            result["timestamp"] = datetime.now()
            result["query"] = query

            # Store result
            st.session_state.workflow_results.append(result)
            st.session_state.current_result = result

            # Update stats
            self.update_system_stats()

            # Display results
            self.display_workflow_results(result)

        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ Workflow failed!")
            st.error(f"Workflow execution failed: {str(e)}")
            st.exception(e)

    def search_similar_solutions(self, query: str):
        """Search for similar solutions"""
        if not self.workflow:
            return

        try:
            with st.spinner("Searching for similar approved solutions..."):
                similar = self.workflow.search_similar_solutions(query, n_results=5)
                st.session_state.similar_solutions = similar

            if similar:
                st.success(f"Found {len(similar)} similar solutions!")
                self.display_similar_solutions(similar)
            else:
                st.info("No similar approved solutions found.")

        except Exception as e:
            st.error(f"Search failed: {str(e)}")

    def display_similar_solutions(self, solutions: List[Dict[str, Any]]):
        """Display similar solutions"""
        if not solutions:
            return

        st.subheader("🔍 Similar Approved Solutions")

        for i, solution in enumerate(solutions, 1):
            content = solution.get("content", "")
            metadata = solution.get("metadata", {})

            # Extract task and code from content
            task_part = ""
            code_part = ""
            if "Task:" in content and "SysML Code:" in content:
                parts = content.split("SysML Code:")
                task_part = parts[0].replace("Task:", "").strip()
                code_part = parts[1].strip() if len(parts) > 1 else ""
            else:
                code_part = content

            created_at = metadata.get("created_at", "N/A")
            iterations = metadata.get("iterations_used", "N/A")
            solution_id = metadata.get("solution_id", f"solution_{i}")

            with st.expander(f"Solution {i} - Created: {created_at}"):
                if task_part:
                    st.write("**Original Task:**", task_part[:200])
                st.write("**Iterations Used:**", iterations)
                st.write("**Solution ID:**", solution_id)

                if code_part:
                    st.code(
                        code_part[:500] + ("..." if len(code_part) > 500 else ""),
                        language="text",
                    )

    def display_workflow_results(self, result: Dict[str, Any]):
        """Display comprehensive workflow results"""
        st.subheader("📋 Workflow Results")

        # Result summary
        success = result.get("success", False)
        if success:
            st.markdown(
                '<div class="success-box">✅ Workflow completed successfully!</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="error-box">❌ Workflow failed!</div>',
                unsafe_allow_html=True,
            )

        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            iterations = result.get("iteration", 0)
            st.metric("Iterations Used", iterations)

        with col2:
            execution_time = result.get("execution_time", 0)
            st.metric("Execution Time", f"{execution_time:.2f}s")

        with col3:
            approval_status = result.get("approval_status", "N/A")
            st.metric("Approval Status", approval_status)

        with col4:
            is_valid = result.get("is_valid", "pending")
            st.metric("Validation Status", is_valid)

        # Generated code
        code = result.get("code", "")
        if code:
            st.subheader("📄 Generated SysML Code")
            st.code(code, language="text")

            # Download button
            st.download_button(
                "💾 Download Code",
                code,
                file_name=f"sysml_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sysml",
                mime="text/plain",
            )

        # Validation history
        if result.get("validation_history"):
            st.subheader("🔍 Validation History")
            for i, validation in enumerate(result["validation_history"], 1):
                success = validation.get("success", False)
                status = "✅ Success" if success else "❌ Failed"

                with st.expander(f"Validation {i}: {status}"):
                    st.write("**Success:**", success)

                    # Show output
                    if validation.get("output"):
                        st.write("**Output:**", validation["output"])

                    # Show errors if present
                    if validation.get("errors"):
                        st.error("**Errors:**")
                        for error in validation["errors"]:
                            error_name = error.get("name", "Error")
                            error_msg = error.get("message", "Unknown error")
                            st.write(f"- {error_name}: {error_msg}")

                    # Show warnings if present
                    if validation.get("warnings"):
                        st.warning("**Warnings:**")
                        for warning in validation["warnings"]:
                            st.write(f"- {warning}")

        # Error details
        if result.get("error") and not success:
            st.subheader("❌ Error Details")
            st.error(result["error"])

    def render_history_tab(self):
        """Render the execution history tab"""
        st.subheader("📜 Execution History")

        if not st.session_state.workflow_results:
            st.info(
                "No execution history available. Run some queries to see results here."
            )
            return

        # Summary statistics
        results = st.session_state.workflow_results
        total_queries = len(results)
        successful_queries = sum(1 for r in results if r.get("success", False))
        avg_iterations = (
            sum(r.get("iteration", 0) for r in results) / total_queries
            if total_queries > 0
            else 0
        )
        avg_execution_time = (
            sum(r.get("execution_time", 0) for r in results) / total_queries
            if total_queries > 0
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            success_rate = (
                (successful_queries / total_queries * 100) if total_queries > 0 else 0
            )
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            st.metric("Avg Iterations", f"{avg_iterations:.1f}")
        with col4:
            st.metric("Avg Execution Time", f"{avg_execution_time:.2f}s")

        # Results timeline
        if len(results) > 1:
            st.subheader("📈 Performance Timeline")

            df = pd.DataFrame(
                [
                    {
                        "timestamp": r.get("timestamp", datetime.now()),
                        "success": r.get("success", False),
                        "iterations": r.get("iteration", 0),
                        "execution_time": r.get("execution_time", 0),
                    }
                    for r in results
                ]
            )

            # Execution time trend
            fig_time = px.line(
                df, x="timestamp", y="execution_time", title="Execution Time Trend"
            )
            st.plotly_chart(fig_time, width="stretch")

        # Individual results
        st.subheader("📋 Individual Results")

        for i, result in enumerate(reversed(results), 1):
            timestamp = result.get("timestamp", datetime.now())
            query = result.get("query", "Unknown query")
            success = result.get("success", False)

            status_icon = "✅" if success else "❌"

            with st.expander(
                f"{status_icon} Query {len(results) - i + 1} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            ):
                st.write("**Query:**", query)
                st.write("**Success:**", "Yes" if success else "No")
                st.write("**Iterations:**", result.get("iteration", 0))
                st.write(
                    "**Execution Time:**", f"{result.get('execution_time', 0):.2f}s"
                )

                code = result.get("code", "")
                if code:
                    st.code(
                        code[:300] + ("..." if len(code) > 300 else ""), language="text"
                    )

    def render_analytics_tab(self):
        """Render the analytics and insights tab"""
        st.subheader("📊 Analytics & Insights")

        if not st.session_state.workflow_results:
            st.info("No data available for analytics. Run some queries first.")
            return

        results = st.session_state.workflow_results

        # Performance analytics
        col1, col2 = st.columns(2)

        with col1:
            # Success rate distribution
            success_counts = {
                "Success": sum(1 for r in results if r.get("success", False)),
                "Failed": sum(1 for r in results if not r.get("success", False)),
            }

            fig_success = go.Figure(
                data=[
                    go.Pie(
                        labels=list(success_counts.keys()),
                        values=list(success_counts.values()),
                    )
                ]
            )
            fig_success.update_layout(title="Success vs Failure Distribution")
            st.plotly_chart(fig_success, width='content')

        with col2:
            # Iterations distribution
            iterations_data = [r.get("iteration", 0) for r in results]
            fig_iter = go.Figure(data=[go.Histogram(x=iterations_data, nbinsx=10)])
            fig_iter.update_layout(
                title="Iterations Distribution",
                xaxis_title="Iterations",
                yaxis_title="Frequency",
            )
            st.plotly_chart(fig_iter, width='content')

        # System performance insights
        st.subheader("🔎 System Insights")

        if st.session_state.system_stats:
            stats = st.session_state.system_stats

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**RAG System Performance**")
                rag_stats = stats.get("rag", {})
                if rag_stats.get("enabled"):
                    total_chunks = rag_stats.get("total_chunks", 0)
                    if total_chunks > 0:
                        st.success(f"✅ RAG Enabled with {total_chunks} chunks")
                    else:
                        st.warning("⚠️ RAG Enabled but database is empty")
                        st.info("💡 Run: python Injestion.py")
                else:
                    st.warning("⚠️ RAG System Disabled")

            with col2:
                st.markdown("**Approved Solutions Database**")
                approved_stats = stats.get("approved_db", {})
                if approved_stats.get("enabled"):
                    st.success(
                        f"✅ Enabled with {approved_stats.get('total_entries', 0)} entries"
                    )
                else:
                    st.warning("⚠️ Approved Solutions DB Disabled")

    def render_about_tab(self):
        """Render the about/help tab"""
        st.subheader("ℹ️ About SysML AI Workflow")

        st.markdown(
            """
        ## 🤖 Enhanced SysML Workflow Assistant
        
        ### ✨ Architecture
        
        **Query Agent → Template Agent → SysML Agent → Human Approval**
        
        1. **Query Agent**: Analyzes and structures user requirements
        2. **Template Agent**: Creates initial SysML template structure
        3. **SysML Agent**: Refines, validates, and enhances code using RAG tool
        4. **Human Approval**: Quality assurance and feedback loop
        
        ### 🔧 Key Features
        
        - **🧠 Multi-Agent Architecture**: Specialized agents for different tasks
        - **🔍 RAG as Tool**: SysML Agent can query knowledge base when needed
        - **💾 Vector Database**: Stores approved solutions for learning
        - **✅ Automated Validation**: Built-in syntax and semantic checking
        - **🔄 Error Correction**: Automatic retry with template regeneration
        - **👥 Human-in-the-Loop**: Manual review and approval process
        - **📊 Analytics**: Track performance and system insights
        
        ### 🚀 Getting Started
        
        1. Configure settings in sidebar
        2. Initialize workflow
        3. Enter SysML modeling request
        4. Review generated code
        5. Approve or provide feedback
        
        ### 📈 Performance Tips
        
        - Enable RAG for better context
        - Populate RAG database with examples
        - Enable approved solutions database
        - Be specific in queries
        - Review analytics for insights
        
        ### 🛠️ Technical Stack
        
        - **Framework**: LangGraph
        - **Models**: Ollama (customizable)
        - **Vector Store**: ChromaDB
        - **Embeddings**: HuggingFace
        - **UI**: Streamlit
        """
        )

    def run(self):
        """Run the Streamlit application"""
        # Render header
        self.render_header()

        # Get configuration from sidebar
        config = self.render_sidebar()

        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🏠 Home", "📜 History", "📊 Analytics", "ℹ️ About"]
        )

        with tab1:
            self.render_system_overview()
            self.render_query_interface()

        with tab2:
            self.render_history_tab()

        with tab3:
            self.render_analytics_tab()

        with tab4:
            self.render_about_tab()


def main():
    """Main application entry point"""
    app = StreamlitSysMLApp()
    app.run()


if __name__ == "__main__":
    main()
