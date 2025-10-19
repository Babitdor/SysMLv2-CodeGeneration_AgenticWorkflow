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
    from AgentWorkflow import EnhancedSysMLWorkflow

    WORKFLOW_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import EnhancedSysMLWorkflow: {e}")
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
</style>
""",
    unsafe_allow_html=True,
)


class StreamlitSysMLApp:
    """Streamlit app wrapper for Enhanced SysML Workflow"""

    def __init__(self):
        self.workflow = st.session_state.get("workflow", None)  # restore if available
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
        st.markdown("---")

    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.header("⚙️ Configuration")

        # Workflow settings
        st.sidebar.subheader("Workflow Settings")

        model_name = st.sidebar.text_input(
            "Model Name",
            value="SysML-V2-llama3.1:latest",
            help="Ollama model name for code generation",
        )

        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.15,
            step=0.05,
            help="Controls randomness in generation (lower = more deterministic)",
        )

        max_iterations = st.sidebar.slider(
            "Max Iterations",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum correction iterations before giving up",
        )

        # RAG and Vector DB settings
        st.sidebar.subheader("Advanced Features")

        enable_rag = st.sidebar.checkbox(
            "Enable RAG System",
            value=True,
            help="Use Retrieval Augmented Generation for better context",
        )

        enable_vector_db = st.sidebar.checkbox(
            "Enable Vector Database",
            value=True,
            help="Store and retrieve similar solutions",
        )

        force_rebuild_rag = st.sidebar.checkbox(
            "Force RAG Rebuild", value=False, help="Rebuild RAG cache (slower startup)"
        )

        # Data paths
        st.sidebar.subheader("Data Paths")
        rag_data_path = st.sidebar.text_input(
            "RAG Data Path", value="./rag_data", help="Path to RAG training data"
        )

        config_path = st.sidebar.text_input(
            "Config Path",
            value="./prompt.yaml",
            help="Path to agent configuration file",
        )

        # Initialize/Reset workflow
        st.sidebar.subheader("Workflow Control")

        if st.sidebar.button("🚀 Initialize Workflow", type="primary"):
            self.initialize_workflow(
                model_name=model_name,
                temperature=temperature,
                enable_rag=enable_rag,
                enable_vector_db=enable_vector_db,
                force_rebuild_rag=force_rebuild_rag,
                rag_data_path=rag_data_path,
                config_path=config_path,
            )

        if st.sidebar.button("🔄 Reset Workflow"):
            self.reset_workflow()

        # System status
        if st.session_state.workflow_initialized:
            st.sidebar.subheader("🟢 System Status")
            st.sidebar.success("Workflow Ready")
        else:
            st.sidebar.subheader("🔴 System Status")
            st.sidebar.error("Workflow Not Initialized")

        return {
            "model_name": model_name,
            "temperature": temperature,
            "max_iterations": max_iterations,
            "enable_rag": enable_rag,
            "enable_vector_db": enable_vector_db,
            "force_rebuild_rag": force_rebuild_rag,
            "rag_data_path": rag_data_path,
            "config_path": config_path,
        }

    def initialize_workflow(self, **kwargs):
        """Initialize the workflow with given parameters"""
        if not WORKFLOW_AVAILABLE:
            st.error("Enhanced SysML Workflow not available. Please check imports.")
            return

        try:
            with st.spinner("Initializing Enhanced SysML Workflow..."):
                workflow = EnhancedSysMLWorkflow(
                    config_path=kwargs["config_path"],
                    model_name=kwargs["model_name"],
                    temperature=kwargs["temperature"],
                    rag_data_path=kwargs["rag_data_path"],
                    enable_vector_storage=kwargs["enable_vector_db"],
                    enable_rag=kwargs["enable_rag"],
                    force_rebuild_rag=kwargs["force_rebuild_rag"],
                )

                st.session_state.workflow = workflow  # <-- store in session
                st.session_state.workflow_initialized = True
                self.workflow = workflow  # also keep in instance

                self.update_system_stats()
                st.success("✅ Workflow initialized successfully!")

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
        st.session_state.workflow_initialized = False
        st.session_state.workflow_results = []
        st.session_state.current_result = None
        st.session_state.system_stats = {}
        st.session_state.similar_solutions = []
        st.success("🔄 Workflow reset successfully!")

    def update_system_stats(self):
        """Update system statistics"""
        if not self.workflow:
            return

        try:
            rag_stats = self.workflow.get_rag_statistics()
            vector_stats = self.workflow.get_vector_db_statistics()

            st.session_state.system_stats = {
                "rag": rag_stats,
                "vector_db": vector_stats,
                "total_results": len(st.session_state.workflow_results),
                "last_updated": datetime.now(),
            }
        except Exception as e:
            st.warning(f"Could not update system stats: {e}")

    def render_system_overview(self):
        """Render system overview and statistics"""
        st.subheader("📊 System Overview")

        if not st.session_state.workflow_initialized:
            # Show helpful initialization guide
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
                """
                )

            with col2:
                st.image(
                    "https://via.placeholder.com/200x150/1f77b4/white?text=SysML+AI",
                    caption="SysML AI Assistant",
                )

            # Show a preview of what will be available
            st.subheader("🔮 Available Features (After Initialization)")

            preview_col1, preview_col2, preview_col3, preview_col4 = st.columns(4)

            with preview_col1:
                st.metric("RAG System", "Ready", help="Context enhancement system")
            with preview_col2:
                st.metric("Vector Database", "Ready", help="Solution storage system")
            with preview_col3:
                st.metric("Total Queries", "0", help="Queries processed")
            with preview_col4:
                st.metric("Success Rate", "N/A", help="Overall success rate")

            return

        # Create metrics columns (when initialized)
        col1, col2, col3, col4 = st.columns(4)

        stats = st.session_state.system_stats

        with col1:
            rag_enabled = stats.get("rag", {}).get("enabled", False)
            rag_docs = stats.get("rag", {}).get("total_documents", 0)
            st.metric(
                "RAG System",
                "Enabled" if rag_enabled else "Disabled",
                f"{rag_docs} documents" if rag_enabled else None,
            )

        with col2:
            vdb_enabled = stats.get("vector_db", {}).get("enabled", False)
            vdb_entries = stats.get("vector_db", {}).get("total_entries", 0)
            st.metric(
                "Vector Database",
                "Enabled" if vdb_enabled else "Disabled",
                f"{vdb_entries} entries" if vdb_enabled else None,
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

    def render_query_interface(self):
        """Render the main query interface"""
        if not st.session_state.workflow_initialized:
            return

        st.subheader("💬 SysML Query Interface")

        # Query input
        query = st.text_area(
            "Enter your SysML modeling request:",
            placeholder="e.g., Create a SysML block definition for an automotive brake system with ABS functionality",
            height=100,
            help="Describe what SysML model you want to generate. Be specific about requirements, constraints, and desired features.",
        )

        # Execution options
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.button(
                "🚀 Generate SysML Code", type="primary", disabled=not query.strip()
            ):
                self.execute_workflow(query.strip())

        with col2:
            max_iterations = st.number_input(
                "Max Iterations", min_value=1, max_value=10, value=5
            )

        with col3:
            search_similar = st.button("🔍 Search Similar", disabled=not query.strip())

        if search_similar:
            self.search_similar_solutions(query.strip())

    def execute_workflow(self, query: str):
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
            result = self.workflow.run(query, max_iterations=5)
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
            with st.spinner("Searching for similar solutions..."):
                similar = self.workflow.search_similar_solutions(query, n_results=5)
                st.session_state.similar_solutions = similar

            if similar:
                st.success(f"Found {len(similar)} similar solutions!")
                self.display_similar_solutions(similar)
            else:
                st.info("No similar solutions found.")

        except Exception as e:
            st.error(f"Search failed: {str(e)}")

    def display_similar_solutions(self, solutions: List[Dict[str, Any]]):
        """Display similar solutions"""
        if not solutions:
            return

        st.subheader("🔍 Similar Solutions")

        for i, solution in enumerate(solutions, 1):
            with st.expander(
                f"Solution {i} - Similarity: {solution.get('similarity_score', 0):.3f}"
            ):
                st.write("**Task:**", solution.get("task", "N/A")[:200] + "...")
                st.write("**Status:**", solution.get("status", "N/A"))

                if solution.get("code"):
                    st.code(
                        solution["code"][:500] + "..."
                        if len(solution["code"]) > 500
                        else solution["code"]
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
            iterations = result.get("iterations", 0)
            st.metric("Iterations", iterations)

        with col2:
            execution_time = result.get("execution_time", 0)
            st.metric("Execution Time", f"{execution_time:.2f}s")

        with col3:
            approval_status = result.get("approval_status", "Unknown")
            st.metric("Approval Status", approval_status)

        with col4:
            if "success_rate" in result:
                success_rate = result["success_rate"] * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")

        # Generated code
        if result.get("final_code"):
            st.subheader("📝 Generated SysML Code")
            st.code(result["final_code"], language="text")

            # Download button
            st.download_button(
                "💾 Download Code",
                result["final_code"],
                file_name=f"sysml_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sysml",
                mime="text/plain",
            )

        # Validation results
        if result.get("validation_history"):
            st.subheader("🔍 Validation History")
            for i, validation in enumerate(result["validation_history"], 1):
                status = "✅ Success" if validation.get("success") else "❌ Failed"
                st.write(f"**Iteration {i}:** {status}")
                if validation.get("error_message"):
                    st.error(f"Error: {validation['error_message']}")

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
            sum(r.get("iterations", 0) for r in results) / total_queries
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
                        "iterations": r.get("iterations", 0),
                        "execution_time": r.get("execution_time", 0),
                    }
                    for r in results
                ]
            )

            # Success rate over time
            fig_success = px.scatter(
                df,
                x="timestamp",
                y="success",
                title="Success Rate Over Time",
                color="success",
            )
            st.plotly_chart(fig_success, use_container_width=True)

            # Execution time trend
            fig_time = px.line(
                df, x="timestamp", y="execution_time", title="Execution Time Trend"
            )
            st.plotly_chart(fig_time, use_container_width=True)

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
                st.write("**Iterations:**", result.get("iterations", 0))
                st.write(
                    "**Execution Time:**", f"{result.get('execution_time', 0):.2f}s"
                )

                if result.get("final_code"):
                    st.code(
                        result["final_code"][:300] + "..."
                        if len(result["final_code"]) > 300
                        else result["final_code"]
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
            st.plotly_chart(fig_success, use_container_width=True)

        with col2:
            # Iterations distribution
            iterations_data = [r.get("iterations", 0) for r in results]
            fig_iter = go.Figure(data=[go.Histogram(x=iterations_data, nbinsx=10)])
            fig_iter.update_layout(
                title="Iterations Distribution",
                xaxis_title="Iterations",
                yaxis_title="Frequency",
            )
            st.plotly_chart(fig_iter, use_container_width=True)

        # System performance insights
        st.subheader("🔍 System Insights")

        if st.session_state.system_stats:
            stats = st.session_state.system_stats

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**RAG System Performance**")
                rag_stats = stats.get("rag", {})
                if rag_stats.get("enabled"):
                    st.success(
                        f"✅ RAG Enabled with {rag_stats.get('total_documents', 0)} documents"
                    )
                    if "error_documents" in rag_stats:
                        st.info(f"📚 Error Solutions: {rag_stats['error_documents']}")
                else:
                    st.warning("⚠️ RAG System Disabled")

            with col2:
                st.markdown("**Vector Database Performance**")
                vdb_stats = stats.get("vector_db", {})
                if vdb_stats.get("enabled"):
                    st.success(
                        f"✅ Vector DB Enabled with {vdb_stats.get('total_entries', 0)} entries"
                    )
                else:
                    st.warning("⚠️ Vector Database Disabled")

        # Recommendations
        st.subheader("💡 Recommendations")

        if len(results) >= 3:
            avg_iterations = sum(r.get("iterations", 0) for r in results) / len(results)
            success_rate = sum(1 for r in results if r.get("success", False)) / len(
                results
            )

            recommendations = []

            if avg_iterations > 3:
                recommendations.append(
                    "🔧 High average iterations detected. Consider improving initial prompts or validation rules."
                )

            if success_rate < 0.7:
                recommendations.append(
                    "⚠️ Success rate is below 70%. Consider enabling RAG or reviewing model configuration."
                )

            if not st.session_state.system_stats.get("rag", {}).get("enabled"):
                recommendations.append(
                    "💡 Enable RAG system for better context and improved results."
                )

            if not st.session_state.system_stats.get("vector_db", {}).get("enabled"):
                recommendations.append(
                    "💾 Enable Vector Database to learn from approved solutions."
                )

            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success(
                    "🎉 System is performing well! No recommendations at this time."
                )
        else:
            st.info("Run more queries to get personalized recommendations.")

    def run(self):
        """Main application runner"""
        self.render_header()

        # Render sidebar
        config = self.render_sidebar()

        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🏠 Main", "📜 History", "📊 Analytics", "ℹ️ About"]
        )

        with tab1:
            self.render_system_overview()
            st.markdown("---")
            self.render_query_interface()

            # Display current results if available
            if st.session_state.current_result:
                st.markdown("---")
                self.display_workflow_results(st.session_state.current_result)

            # Display similar solutions if available
            if st.session_state.similar_solutions:
                st.markdown("---")
                self.display_similar_solutions(st.session_state.similar_solutions)

        with tab2:
            self.render_history_tab()

        with tab3:
            self.render_analytics_tab()

        with tab4:
            self.render_about_tab()

    def render_about_tab(self):
        """Render the about/help tab"""
        st.subheader("ℹ️ About SysML AI Workflow")

        st.markdown(
            """
        ## 🤖 Enhanced SysML Workflow Assistant
        
        This application provides an intelligent workflow for generating SysML (Systems Modeling Language) code using AI agents with advanced features:
        
        ### ✨ Key Features
        
        - **🧠 AI-Powered Code Generation**: Uses fine-tuned language models for SysML code generation
        - **🔍 RAG System**: Retrieval Augmented Generation for better context and examples
        - **💾 Vector Database**: Stores and retrieves similar approved solutions
        - **✅ Automated Validation**: Multi-step validation with error correction
        - **👥 Human-in-the-Loop**: Human approval process for quality assurance
        - **📊 Analytics & Insights**: Performance tracking and system optimization
        
        ### 🔧 How It Works
        
        1. **Query Processing**: Your request is analyzed and enhanced with context
        2. **RAG Enhancement**: Relevant examples and patterns are retrieved
        3. **Code Generation**: AI generates SysML code based on enhanced context
        4. **Validation**: Automated syntax and semantic validation
        5. **Error Correction**: Automatic fixing of detected issues
        6. **Human Approval**: Optional human review and approval
        7. **Knowledge Update**: Approved solutions are stored for future use
        
        ### 🚀 Getting Started
        
        1. Configure the system in the sidebar
        2. Click "Initialize Workflow" to set up the system
        3. Enter your SysML modeling request in the main interface
        4. Click "Generate SysML Code" to start the workflow
        5. Review results and download generated code
        
        ### 📈 Performance Tips
        
        - Enable RAG for better context and examples
        - Enable Vector Database to learn from previous solutions
        - Be specific in your queries for better results
        - Review the analytics tab for system insights
        
        ### 🛠️ Technical Details
        
        - **Framework**: LangGraph for workflow orchestration
        - **Models**: Ollama-based language models
        - **Vector Store**: ChromaDB for similarity search
        - **Embeddings**: HuggingFace embeddings
        - **UI**: Streamlit for interactive interface
        
        ### 📞 Support
        
        For technical issues or questions, please refer to the system documentation or contact the development team.
        """
        )

        # System information
        st.subheader("🖥️ System Information")

        system_info = {
            "Workflow Available": "Yes" if WORKFLOW_AVAILABLE else "No",
            "Session ID": st.session_state.get("session_id", "N/A"),
            "Total Queries": len(st.session_state.workflow_results),
            "Workflow Initialized": (
                "Yes" if st.session_state.workflow_initialized else "No"
            ),
        }

        for key, value in system_info.items():
            st.write(f"**{key}:** {value}")


def main():
    """Main application entry point"""
    app = StreamlitSysMLApp()
    app.run()


if __name__ == "__main__":
    main()
