import os
import logging
from typing import List, Optional
from langchain.schema import Document
# Simplified text-based search without vector embeddings initially

logger = logging.getLogger(__name__)

class VectorStore:
    """Simple text-based document store"""
    
    def __init__(self):
        """Initialize document store"""
        self.documents = []
        self.indexed = False
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for search"""
        import re
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        return text
    
    def create_index(self, documents: List[Document]):
        """
        Create document index from documents
        
        Args:
            documents: List of Document objects to index
        """
        try:
            if not documents:
                raise ValueError("No documents provided for indexing")
            
            logger.info(f"Creating document index for {len(documents)} documents...")
            
            # Filter out empty documents and preprocess
            valid_docs = []
            for doc in documents:
                if doc.page_content.strip():
                    # Add preprocessed version for search
                    doc.metadata['search_text'] = self._preprocess_text(doc.page_content)
                    valid_docs.append(doc)
            
            self.documents = valid_docs
            self.indexed = True
            logger.info(f"Indexed {len(valid_docs)} valid documents")
            
        except Exception as e:
            logger.error(f"Error creating document index: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents using text matching
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        try:
            if not self.indexed:
                raise ValueError("Document store not initialized")
            
            # Preprocess query
            query_processed = self._preprocess_text(query)
            query_words = set(query_processed.split())
            
            # Score documents based on word overlap
            scored_docs = []
            for doc in self.documents:
                search_text = doc.metadata.get('search_text', '')
                doc_words = set(search_text.split())
                
                # Calculate similarity score (Jaccard similarity)
                intersection = len(query_words.intersection(doc_words))
                union = len(query_words.union(doc_words))
                score = intersection / union if union > 0 else 0
                
                if score > 0:
                    scored_docs.append((doc, score))
            
            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            docs = [doc for doc, score in scored_docs[:k]]
            
            logger.info(f"Found {len(docs)} similar documents for query: {query[:50]}...")
            return docs
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for similar documents with relevance scores
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if not self.indexed:
                raise ValueError("Document store not initialized")
            
            # Preprocess query
            query_processed = self._preprocess_text(query)
            query_words = set(query_processed.split())
            
            # Score documents based on word overlap
            scored_docs = []
            for doc in self.documents:
                search_text = doc.metadata.get('search_text', '')
                doc_words = set(search_text.split())
                
                # Calculate similarity score
                intersection = len(query_words.intersection(doc_words))
                union = len(query_words.union(doc_words))
                score = intersection / union if union > 0 else 0
                
                if score > 0:
                    scored_docs.append((doc, score))
            
            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            docs_with_scores = scored_docs[:k]
            
            logger.info(f"Found {len(docs_with_scores)} documents with scores for query: {query[:50]}...")
            return docs_with_scores
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {str(e)}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """
        Add new documents to existing document store
        
        Args:
            documents: List of Document objects to add
        """
        try:
            if not self.indexed:
                raise ValueError("Document store not initialized")
            
            valid_docs = []
            for doc in documents:
                if doc.page_content.strip():
                    doc.metadata['search_text'] = self._preprocess_text(doc.page_content)
                    valid_docs.append(doc)
            
            if valid_docs:
                self.documents.extend(valid_docs)
                logger.info(f"Added {len(valid_docs)} documents to document store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def save_local(self, folder_path: str):
        """Save document store to local directory (placeholder)"""
        try:
            if not self.indexed:
                raise ValueError("Document store not initialized")
            
            # For now, just log that we would save
            logger.info(f"Document store would be saved to {folder_path}")
            
        except Exception as e:
            logger.error(f"Error saving document store: {str(e)}")
            raise
    
    def load_local(self, folder_path: str):
        """Load document store from local directory (placeholder)"""
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Document store folder not found: {folder_path}")
            
            # For now, just log that we would load
            logger.info(f"Document store would be loaded from {folder_path}")
            
        except Exception as e:
            logger.error(f"Error loading document store: {str(e)}")
            raise