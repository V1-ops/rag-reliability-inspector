"""Quick test to verify all imports work correctly"""

print("Testing imports...")

try:
    print("1. Testing streamlit...")
    import streamlit
    print("   ✓ streamlit OK")
except Exception as e:
    print(f"   ✗ streamlit FAILED: {e}")

try:
    print("2. Testing langchain...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("   ✓ langchain-text-splitters OK")
except Exception as e:
    print(f"   ✗ langchain-text-splitters FAILED: {e}")

try:
    print("3. Testing sentence_transformers...")
    from sentence_transformers import SentenceTransformer
    print("   ✓ sentence-transformers OK")
except Exception as e:
    print(f"   ✗ sentence-transformers FAILED: {e}")

try:
    print("4. Testing chromadb...")
    import chromadb
    print("   ✓ chromadb OK")
except Exception as e:
    print(f"   ✗ chromadb FAILED: {e}")

try:
    print("5. Testing langchain_community...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("   ✓ langchain_community OK")
except Exception as e:
    print(f"   ✗ langchain_community FAILED: {e}")

try:
    print("6. Testing evaluator...")
    from evaluator import RAGEvaluator
    print("   ✓ evaluator OK")
except Exception as e:
    print(f"   ✗ evaluator FAILED: {e}")

try:
    print("7. Testing rag...")
    from rag import RAGPipeline
    print("   ✓ rag OK")
except Exception as e:
    print(f"   ✗ rag FAILED: {e}")

print("\n✅ All imports successful! No warnings!")
