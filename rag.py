"""
RAG Pipeline Module
Handles document loading, chunking, embedding, and retrieval.
"""

from typing import List, Dict, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
import tempfile


class RAGPipeline:
    """RAG Pipeline for document processing and question answering."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        hf_token: str = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model: HuggingFace embedding model name
            llm_model: HuggingFace LLM model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            hf_token: HuggingFace API token
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hf_token = hf_token
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LLM
        if hf_token:
            self.llm_endpoint = HuggingFaceEndpoint(
                repo_id=llm_model,
                huggingfacehub_api_token=hf_token,
                temperature=0.7,
                max_new_tokens=512
            )
            self.llm = ChatHuggingFace(llm=self.llm_endpoint)
        else:
            self.llm = None
        
        self.vectorstore = None
        self.retriever = None
        
    def load_documents(self, file_path: str, file_type: str = "pdf") -> List:
        """
        Load documents from file.
        
        Args:
            file_path: Path to the document file
            file_type: Type of file ('pdf' or 'txt')
            
        Returns:
            List of loaded documents
        """
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return loader.load()
    
    def chunk_documents(self, documents: List) -> List:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        return text_splitter.split_documents(documents)
    
    def create_vectorstore(self, chunks: List) -> None:
        """
        Create vector store from document chunks.
        
        Args:
            chunks: List of document chunks
        """
        # Clear existing vectorstore if it exists
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except:
                pass
            self.vectorstore = None
        
        # Create fresh vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="rag_collection"
        )
    
    def setup_retriever(self, k: int = 3) -> None:
        """
        Setup retriever with top-k documents.
        
        Args:
            k: Number of documents to retrieve
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def retrieve_with_scores(self, query: str, k: int = 3) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (documents, scores)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        # Get documents with scores
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Separate documents and scores
        docs = [doc for doc, score in results]
        # Chroma returns distance (lower is better), convert to similarity
        scores = [1 / (1 + score) for doc, score in results]
        
        return docs, scores
    
    def generate_answer(self, query: str, k: int = 3, relevance_threshold: float = 0.35) -> Dict:
        """
        Generate answer using RAG pipeline.
        
        Args:
            query: Question to answer
            k: Number of documents to retrieve
            relevance_threshold: Minimum average similarity score to consider question in-scope
            
        Returns:
            Dictionary with answer and retrieved documents
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Provide HuggingFace token.")
        
        # Get documents with scores
        docs, scores = self.retrieve_with_scores(query, k=k)
        
        # Check if question is out of scope (low relevance)
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score < relevance_threshold:
            answer = "⚠️ I cannot find relevant information in the provided document to answer your question. Please ask something within the scope of the uploaded PDF/document."
            return {
                "answer": answer,
                "retrieved_docs": docs,
                "retrieval_scores": scores,
                "context": "",
                "out_of_scope": True
            }
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer with instruction to stay grounded
        prompt = f"""Based on the following context, answer the question. If the context does not contain enough information to answer the question, say "I cannot find relevant information in the document to answer this question."

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        return {
            "answer": answer,
            "retrieved_docs": docs,
            "retrieval_scores": scores,
            "context": context,
            "out_of_scope": False
        }
    
    def process_file(self, file_path: str, file_type: str = "pdf", k: int = 3) -> None:
        """
        Process uploaded file and setup RAG pipeline.
        
        Args:
            file_path: Path to the file
            file_type: Type of file ('pdf' or 'txt')
            k: Number of documents to retrieve
        """
        # Load and chunk documents
        documents = self.load_documents(file_path, file_type)
        chunks = self.chunk_documents(documents)
        
        # Create vector store
        self.create_vectorstore(chunks)
        
        # Setup retriever
        self.setup_retriever(k=k)
