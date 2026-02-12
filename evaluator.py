"""
RAG Evaluation Module
Evaluates RAG performance: retrieval confidence, grounding score, failure diagnosis.
"""

from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re


class RAGEvaluator:
    """Evaluator for RAG pipeline reliability."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize evaluator.
        
        Args:
            embedding_model: Model for sentence embeddings
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Thresholds
        self.retrieval_high_threshold = 0.7
        self.retrieval_medium_threshold = 0.4
        self.grounding_threshold = 0.5
        
    def compute_retrieval_confidence(self, scores: List[float]) -> Dict:
        """
        Compute retrieval confidence from similarity scores.
        
        Args:
            scores: List of similarity scores
            
        Returns:
            Dictionary with avg_score and confidence_level
        """
        if not scores:
            return {
                "avg_score": 0.0,
                "confidence_level": "Low",
                "scores": []
            }
        
        avg_score = np.mean(scores)
        
        # Classify confidence level
        if avg_score >= self.retrieval_high_threshold:
            confidence_level = "High"
        elif avg_score >= self.retrieval_medium_threshold:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        return {
            "avg_score": float(avg_score),
            "confidence_level": confidence_level,
            "scores": [float(s) for s in scores]
        }
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def compute_grounding_score(
        self, 
        answer: str, 
        retrieved_docs: List, 
        threshold: float = None
    ) -> Dict:
        """
        Compute grounding score by comparing answer sentences with retrieved docs.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved document chunks
            threshold: Similarity threshold for support (default: self.grounding_threshold)
            
        Returns:
            Dictionary with grounding metrics
        """
        if threshold is None:
            threshold = self.grounding_threshold
        
        # Split answer into sentences
        answer_sentences = self.split_into_sentences(answer)
        
        if not answer_sentences:
            return {
                "grounding_score": 0.0,
                "supported_sentences": 0,
                "total_sentences": 0,
                "unsupported_sentences": [],
                "sentence_scores": []
            }
        
        # Get context from retrieved docs
        context_texts = [doc.page_content for doc in retrieved_docs]
        
        if not context_texts:
            return {
                "grounding_score": 0.0,
                "supported_sentences": 0,
                "total_sentences": len(answer_sentences),
                "unsupported_sentences": answer_sentences,
                "sentence_scores": [0.0] * len(answer_sentences)
            }
        
        # Encode answer sentences and context
        answer_embeddings = self.embedding_model.encode(answer_sentences, convert_to_tensor=True)
        context_embeddings = self.embedding_model.encode(context_texts, convert_to_tensor=True)
        
        # Compute similarity for each sentence
        supported_count = 0
        unsupported_sentences = []
        sentence_scores = []
        
        for i, sentence in enumerate(answer_sentences):
            # Get max similarity with any context chunk
            similarities = util.cos_sim(answer_embeddings[i], context_embeddings)
            max_similarity = float(similarities.max())
            sentence_scores.append(max_similarity)
            
            if max_similarity >= threshold:
                supported_count += 1
            else:
                unsupported_sentences.append(sentence)
        
        grounding_score = supported_count / len(answer_sentences)
        
        return {
            "grounding_score": float(grounding_score),
            "supported_sentences": supported_count,
            "total_sentences": len(answer_sentences),
            "unsupported_sentences": unsupported_sentences,
            "sentence_scores": sentence_scores
        }
    
    def classify_failure(
        self, 
        retrieval_confidence: Dict, 
        grounding_metrics: Dict
    ) -> Dict:
        """
        Classify RAG failure type based on retrieval and grounding scores.
        
        Args:
            retrieval_confidence: Retrieval confidence metrics
            grounding_metrics: Grounding score metrics
            
        Returns:
            Dictionary with failure classification
        """
        retrieval_level = retrieval_confidence["confidence_level"]
        grounding_score = grounding_metrics["grounding_score"]
        
        # Classification logic
        if retrieval_level == "Low":
            diagnosis = "Retrieval Failure"
            severity = "High"
            description = "The retriever failed to find relevant documents for the query."
        elif retrieval_level == "High" and grounding_score < 0.5:
            diagnosis = "Hallucination Risk"
            severity = "High"
            description = "Retrieved documents are relevant, but the answer contains unsupported claims."
        elif retrieval_level == "Medium":
            diagnosis = "Weak Context"
            severity = "Medium"
            description = "Retrieved documents have moderate relevance. Answer quality may be limited."
        else:
            diagnosis = "Healthy RAG"
            severity = "Low"
            description = "Good retrieval and grounding. The system is performing well."
        
        return {
            "diagnosis": diagnosis,
            "severity": severity,
            "description": description,
            "retrieval_level": retrieval_level,
            "grounding_score": grounding_score
        }
    
    def suggest_fixes(self, failure_classification: Dict, current_k: int = 3) -> List[str]:
        """
        Suggest fixes based on failure classification.
        
        Args:
            failure_classification: Failure classification result
            current_k: Current k value for retrieval
            
        Returns:
            List of fix suggestions
        """
        diagnosis = failure_classification["diagnosis"]
        suggestions = []
        
        if diagnosis == "Retrieval Failure":
            suggestions.append(f"🔧 Increase k from {current_k} to {current_k + 2} to retrieve more documents")
            suggestions.append("🔧 Reduce chunk size to create more granular chunks (e.g., 300-400)")
            suggestions.append("🔧 Try a different embedding model (e.g., 'all-mpnet-base-v2')")
            suggestions.append("🔧 Verify the query is relevant to the uploaded documents")
            
        elif diagnosis == "Hallucination Risk":
            suggestions.append("🔧 Lower LLM temperature to reduce creativity (e.g., 0.3-0.5)")
            suggestions.append("🔧 Add explicit instruction: 'Only use information from the context'")
            suggestions.append("🔧 Increase chunk size to provide more context per document")
            suggestions.append("🔧 Review the prompt template for clarity")
            
        elif diagnosis == "Weak Context":
            suggestions.append(f"🔧 Increase k from {current_k} to {current_k + 1} for more context")
            suggestions.append("🔧 Adjust chunk overlap to 10-20% of chunk size")
            suggestions.append("🔧 Tune chunk size based on document structure")
            suggestions.append("🔧 Consider using a hybrid retrieval method")
            
        else:  # Healthy RAG
            suggestions.append("✅ System is performing well!")
            suggestions.append("💡 Monitor grounding scores over multiple queries")
            suggestions.append("💡 Consider A/B testing different chunk sizes")
        
        return suggestions
    
    def evaluate_rag(
        self, 
        answer: str, 
        retrieved_docs: List, 
        retrieval_scores: List[float],
        current_k: int = 3
    ) -> Dict:
        """
        Complete RAG evaluation pipeline.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents
            retrieval_scores: Retrieval similarity scores
            current_k: Current k value
            
        Returns:
            Complete evaluation report
        """
        # Compute retrieval confidence
        retrieval_confidence = self.compute_retrieval_confidence(retrieval_scores)
        
        # Compute grounding score
        grounding_metrics = self.compute_grounding_score(answer, retrieved_docs)
        
        # Classify failure
        failure_classification = self.classify_failure(
            retrieval_confidence, 
            grounding_metrics
        )
        
        # Get fix suggestions
        fix_suggestions = self.suggest_fixes(failure_classification, current_k)
        
        return {
            "retrieval_confidence": retrieval_confidence,
            "grounding_metrics": grounding_metrics,
            "failure_classification": failure_classification,
            "fix_suggestions": fix_suggestions
        }
