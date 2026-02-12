# 🔍 RAG Reliability Inspector

A beginner-friendly but production-style RAG evaluation system that helps you understand and improve your Retrieval-Augmented Generation (RAG) pipelines.

## 🎯 What This Project Does

Unlike typical RAG chatbots, this project focuses on **evaluating RAG reliability**. It provides:

- **Retrieval Confidence Score**: How relevant are the retrieved documents?
- **Grounding Score**: Is the generated answer supported by the retrieved context?
- **Failure Classification**: Diagnoses issues (Retrieval Failure, Hallucination Risk, Weak Context, or Healthy RAG)
- **Fix Recommendations**: Actionable suggestions to improve your RAG pipeline

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit**: Interactive web UI
- **LangChain**: RAG framework
- **HuggingFace**: LLM inference (ChatHuggingFace)
- **SentenceTransformers**: Embeddings
- **Chroma**: Vector database

## 📁 Project Structure

```
Agentic_AutoML/
├── rag.py          # RAG pipeline (load, chunk, embed, retrieve, generate)
├── evaluator.py    # Evaluation logic (confidence, grounding, diagnosis, fixes)
├── app.py          # Streamlit UI
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Install Dependencies

```powershell
# Activate your virtual environment (if using one)
.venv\Scripts\Activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure HuggingFace API Token

**Option A: Use .env file (Recommended)**

Create or edit the `.env` file in the project root:

```bash
HF_TOKEN=your_token_here
```

The token will be automatically loaded when you run the app.

**Option B: Enter manually in the UI**

1. Go to https://huggingface.co/settings/tokens
2. Create a new access token (read permissions are sufficient)
3. Enter it in the sidebar when running the app

### 3. Run the Application

**Recommended (uses .venv, no warnings):**
```powershell
# Using the run script
.\run.ps1

# OR manually
.venv\Scripts\python.exe -m streamlit run app.py
```

**Alternative (may show numpy warnings from base environment):**
```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### Step 1: Configure Settings

In the sidebar, configure:
- **HuggingFace API Token**: Required for answer generation
- **Chunk Size**: Size of text chunks (default: 500)
- **Chunk Overlap**: Overlap between chunks (default: 50)
- **Top-K Documents**: Number of documents to retrieve (default: 3)

### Step 2: Upload Documents

1. Go to the **"Upload & Process"** tab
2. Upload a PDF or TXT file
3. Click **"Process Document"**
4. Wait for the document to be processed and embedded

### Step 3: Query & Evaluate

1. Go to the **"Query & Evaluate"** tab
2. Enter your question
3. Click **"Generate Answer & Evaluate"**
4. Review the results:
   - **Generated Answer**: The RAG system's response
   - **Retrieved Documents**: Top-K relevant chunks with scores
   - **Retrieval Confidence**: How good the retrieval was
   - **Grounding Score**: How well the answer is supported
   - **Failure Diagnosis**: Classification of any issues
   - **Suggested Fixes**: Actionable recommendations

## 🧪 Understanding the Metrics

### Retrieval Confidence

- **High (≥0.7)**: Retrieved documents are very relevant
- **Medium (0.4-0.7)**: Moderate relevance
- **Low (<0.4)**: Poor retrieval quality

### Grounding Score

Percentage of generated answer sentences that are supported by the retrieved context. Higher is better.

### Failure Classifications

| Diagnosis | Cause | Severity |
|-----------|-------|----------|
| **Retrieval Failure** | Retriever didn't find relevant docs | High |
| **Hallucination Risk** | Good retrieval but unsupported claims in answer | High |
| **Weak Context** | Moderate retrieval quality | Medium |
| **Healthy RAG** | Good retrieval and grounding | Low |

## 💡 Example Use Cases

### 1. Testing Different Chunk Sizes

- Upload a document with chunk size 300
- Query and note the scores
- Process again with chunk size 700
- Compare retrieval and grounding scores

### 2. Detecting Hallucinations

If your RAG shows:
- High retrieval confidence
- Low grounding score
- Diagnosis: "Hallucination Risk"

The LLM is adding information not in the context!

### 3. Improving Poor Retrieval

If you get "Retrieval Failure":
- Increase k (more documents)
- Reduce chunk size (more granular)
- Try different embeddings

## 🔧 Customization Tips

### Change Embedding Model

In `rag.py`, modify:
```python
embedding_model="sentence-transformers/all-mpnet-base-v2"  # More powerful
```

### Change LLM Model

In `rag.py`, modify:
```python
llm_model="meta-llama/Llama-2-7b-chat-hf"  # Different model
```

### Adjust Thresholds

In `evaluator.py`, modify:
```python
self.retrieval_high_threshold = 0.8  # Stricter high threshold
self.grounding_threshold = 0.6       # Stricter grounding threshold
```

## 🐛 Troubleshooting

### "Error initializing pipeline"

- Check your internet connection (models download on first run)
- Ensure you have enough disk space (~500MB for models)

### "Error generating answer"

- Verify your HuggingFace token is valid
- Check if the model `mistralai/Mistral-7B-Instruct-v0.2` is accessible
- Try a different model if you encounter rate limits

### "Out of memory"

- Use a smaller chunk size
- Reduce the number of documents
- Process smaller files

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [HuggingFace Inference](https://huggingface.co/docs/huggingface_hub/guides/inference)
- [Chroma Vector DB](https://docs.trychroma.com/)

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and improve this project! Some ideas:
- Add support for multiple file uploads
- Implement hybrid search (keyword + semantic)
- Add more evaluation metrics (faithfulness, relevance)
- Support for more document formats (DOCX, HTML)

---

Built with ❤️ for learning and production-ready RAG evaluation
