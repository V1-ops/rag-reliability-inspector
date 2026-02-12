"""
RAG Reliability Inspector - Streamlit App
A production-style RAG evaluation system.
"""

import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from rag import RAGPipeline
from evaluator import RAGEvaluator

# Load environment variables from .env file
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="RAG Reliability Inspector",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "evaluator" not in st.session_state:
    st.session_state.evaluator = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None


def init_rag_pipeline(hf_token, chunk_size, chunk_overlap):
    """Initialize RAG pipeline with configuration."""
    try:
        st.session_state.rag_pipeline = RAGPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            hf_token=hf_token if hf_token else None
        )
        st.session_state.evaluator = RAGEvaluator()
        return True
    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        return False


def process_uploaded_file(uploaded_file, file_type, chunk_size, chunk_overlap, k, hf_token):
    """Process uploaded document."""
    with st.spinner("Processing document..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Initialize pipeline
            if not init_rag_pipeline(hf_token, chunk_size, chunk_overlap):
                return False
            
            # Process file
            st.session_state.rag_pipeline.process_file(tmp_file_path, file_type, k)
            st.session_state.documents_loaded = True
            
            # Clean up
            os.unlink(tmp_file_path)
            
            st.success("✅ Document processed successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return False


def display_retrieval_panel(retrieval_confidence):
    """Display retrieval confidence panel."""
    st.subheader("📊 Retrieval Confidence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display average score
        avg_score = retrieval_confidence["avg_score"]
        st.metric("Average Similarity Score", f"{avg_score:.3f}")
        
    with col2:
        # Display confidence level with color coding
        confidence_level = retrieval_confidence["confidence_level"]
        color_map = {
            "High": "🟢",
            "Medium": "🟡",
            "Low": "🔴"
        }
        st.metric("Confidence Level", f"{color_map.get(confidence_level, '')} {confidence_level}")
    
    # Display individual scores
    st.write("**Individual Document Scores:**")
    scores = retrieval_confidence["scores"]
    for i, score in enumerate(scores, 1):
        st.progress(score, text=f"Document {i}: {score:.3f}")


def display_grounding_panel(grounding_metrics):
    """Display grounding score panel."""
    st.subheader("🎯 Grounding Score")
    
    col1, col2 = st.columns(2)
    
    with col1:
        grounding_score = grounding_metrics["grounding_score"]
        st.metric("Grounding Score", f"{grounding_score:.2%}")
        
    with col2:
        supported = grounding_metrics["supported_sentences"]
        total = grounding_metrics["total_sentences"]
        st.metric("Supported Sentences", f"{supported} / {total}")
    
    # Display unsupported sentences
    if grounding_metrics["unsupported_sentences"]:
        st.warning("**⚠️ Potentially Unsupported Claims:**")
        for sentence in grounding_metrics["unsupported_sentences"]:
            st.write(f"- {sentence}")
    else:
        st.success("✅ All sentences are well-grounded in the context!")


def display_failure_panel(failure_classification):
    """Display failure diagnosis panel."""
    st.subheader("🔍 Failure Diagnosis")
    
    diagnosis = failure_classification["diagnosis"]
    severity = failure_classification["severity"]
    description = failure_classification["description"]
    
    # Color code by severity
    severity_colors = {
        "High": "🔴",
        "Medium": "🟡",
        "Low": "🟢"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Diagnosis", f"{severity_colors.get(severity, '')} {diagnosis}")
        
    with col2:
        st.metric("Severity", severity)
    
    st.info(f"**Description:** {description}")


def display_fixes_panel(fix_suggestions):
    """Display fix suggestions panel."""
    st.subheader("💡 Suggested Fixes")
    
    for suggestion in fix_suggestions:
        st.write(suggestion)


def main():
    """Main application."""
    
    # Header
    st.title("🔍 RAG Reliability Inspector")
    st.markdown("""
    An evaluation system for Retrieval-Augmented Generation (RAG) pipelines.
    Upload documents, ask questions, and get detailed reliability metrics.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # HuggingFace token - load from .env as default
    default_token = os.getenv("HF_TOKEN", "")
    hf_token = st.sidebar.text_input(
        "HuggingFace API Token",
        value=default_token,
        type="password",
        help="Required for answer generation. Auto-loaded from .env or enter manually"
    )
    
    # RAG parameters
    st.sidebar.subheader("RAG Parameters")
    chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 500, 50)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50, 10)
    k = st.sidebar.slider("Top-K Documents", 1, 10, 3)
    
    # Main content
    tab1, tab2 = st.tabs(["📁 Upload & Process", "❓ Query & Evaluate"])
    
    # Tab 1: Upload documents
    with tab1:
        st.header("Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF or TXT file",
            type=["pdf", "txt"],
            help="Upload a document to build the knowledge base"
        )
        
        if uploaded_file:
            file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "txt"
            
            if st.button("Process Document", type="primary"):
                success = process_uploaded_file(
                    uploaded_file, 
                    file_type, 
                    chunk_size, 
                    chunk_overlap, 
                    k,
                    hf_token
                )
                
        if st.session_state.documents_loaded:
            st.success("✅ Documents are loaded and ready for querying!")
    
    # Tab 2: Query and evaluate
    with tab2:
        st.header("Query & Evaluate")
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please upload and process a document first!")
        else:
            # Query input
            query = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?"
            )
            
            if st.button("Generate Answer & Evaluate", type="primary"):
                if not query:
                    st.warning("Please enter a question!")
                elif not hf_token:
                    st.warning("Please provide a HuggingFace API token in the sidebar!")
                else:
                    with st.spinner("Generating answer and evaluating..."):
                        try:
                            # Generate answer
                            result = st.session_state.rag_pipeline.generate_answer(query, k=k)
                            
                            # Evaluate
                            evaluation = st.session_state.evaluator.evaluate_rag(
                                answer=result["answer"],
                                retrieved_docs=result["retrieved_docs"],
                                retrieval_scores=result["retrieval_scores"],
                                current_k=k
                            )
                            
                            st.session_state.evaluation_results = {
                                "result": result,
                                "evaluation": evaluation
                            }
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            # Display results
            if st.session_state.evaluation_results:
                result = st.session_state.evaluation_results["result"]
                evaluation = st.session_state.evaluation_results["evaluation"]
                
                # Check if question is out of scope
                if result.get("out_of_scope", False):
                    st.warning("### ⚠️ Question Out of Scope")
                    st.info(result["answer"])
                    st.divider()
                    # Still show retrieved docs to help user understand
                    st.header("📚 Retrieved Documents (Low Relevance)")
                    for i, doc in enumerate(result["retrieved_docs"], 1):
                        with st.expander(f"Document {i} (Score: {result['retrieval_scores'][i-1]:.3f})"):
                            st.write(doc.page_content)
                else:
                    # Answer section
                    st.header("💬 Generated Answer")
                    st.write(result["answer"])
                    
                    st.divider()
                    
                    # Retrieved documents
                    st.header("📚 Retrieved Documents")
                    for i, doc in enumerate(result["retrieved_docs"], 1):
                        with st.expander(f"Document {i} (Score: {result['retrieval_scores'][i-1]:.3f})"):
                            st.write(doc.page_content)
                    
                    st.divider()
                    
                    # Evaluation panels
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_retrieval_panel(evaluation["retrieval_confidence"])
                        st.divider()
                        display_failure_panel(evaluation["failure_classification"])
                    
                    with col2:
                        display_grounding_panel(evaluation["grounding_metrics"])
                        st.divider()
                        display_fixes_panel(evaluation["fix_suggestions"])


if __name__ == "__main__":
    main()
