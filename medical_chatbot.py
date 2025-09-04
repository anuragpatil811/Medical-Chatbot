import os
import logging
from typing import Dict, List
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalChatbot:
    """Medical chatbot using RAG pipeline with Gale Encyclopedia of Medicine"""
    
    def __init__(self):
        """Initialize the medical chatbot"""
        self.pdf_path = "attached_assets/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
        self.vector_store = None
        self.rag_pipeline = None
        self._initialize()
    
    def _initialize(self):
        """Initialize all components"""
        try:
            # Process PDF
            logger.info("Processing PDF...")
            pdf_processor = PDFProcessor()
            documents = pdf_processor.process_pdf(self.pdf_path)
            logger.info(f"Processed {len(documents)} document chunks")
            
            # Create vector store
            logger.info("Creating vector store...")
            self.vector_store = VectorStore()
            self.vector_store.create_index(documents)
            logger.info("Vector store created successfully")
            
            # Initialize RAG pipeline
            logger.info("Initializing RAG pipeline...")
            self.rag_pipeline = RAGPipeline(self.vector_store)
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {str(e)}")
            raise
    
    def get_response(self, query: str) -> Dict[str, any]:
        """
        Get response for user query
        
        Args:
            query: User's medical question
            
        Returns:
            Dictionary containing answer and sources
        """
        try:
            if not self.rag_pipeline:
                raise Exception("RAG pipeline not initialized")
            
            response = self.rag_pipeline.get_response(query)
            return response
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": []
            }
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[str]:
        """Get similar documents for a query"""
        try:
            if not self.vector_store:
                return []
            
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
            
        except Exception as e:
            logger.error(f"Error getting similar documents: {str(e)}")
            return []