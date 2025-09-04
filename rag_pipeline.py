import os
from dotenv import load_dotenv
import logging
from typing import Dict, List
import requests
import json
from langchain.schema import Document

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for medical chatbot"""
    
    def __init__(self, vector_store):
        """
        Initialize RAG pipeline
        
        Args:
            vector_store: Initialized VectorStore instance
        """
        self.vector_store = vector_store
        self.hf_api_key = os.getenv("HF_API_KEY")
        if not self.hf_api_key:
            logger.warning("HF_API_KEY not found in environment. Please set it in the .env file.")
            raise ValueError("Hugging Face API key is required.")
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        
        # Test API connection
        self._test_api_connection()
    
    def _test_api_connection(self):
        """Test connection to Hugging Face API"""
        try:
            test_payload = {
                "inputs": "Hello, this is a test.",
                "parameters": {"max_new_tokens": 10}
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("Successfully connected to Hugging Face API")
            else:
                logger.warning(f"API test returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.warning(f"API connection test failed: {str(e)}")
    
    def get_response(self, query: str) -> Dict[str, any]:
        """
        Get response using RAG pipeline
        
        Args:
            query: User's medical question
            
        Returns:
            Dictionary containing answer and sources
        """
        try:
            # Step 1: Retrieve relevant documents
            relevant_docs = self._retrieve_documents(query)
            
            # Step 2: Generate response using retrieved context
            answer = self._generate_response(query, relevant_docs)
            
            # Step 3: Extract sources
            sources = self._extract_sources(relevant_docs)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": []
            }
    
    def _retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents from vector store"""
        try:
            # Get similar documents
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Filter documents by relevance (simple keyword matching as additional filter)
            filtered_docs = []
            query_words = set(query.lower().split())
            
            for doc in docs:
                doc_words = set(doc.page_content.lower().split())
                # Check if there's some overlap between query and document
                if query_words.intersection(doc_words) or len(filtered_docs) < 2:
                    filtered_docs.append(doc)
            
            logger.info(f"Retrieved {len(filtered_docs)} relevant documents")
            return filtered_docs[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def _generate_response(self, query: str, context_docs: List[Document]) -> str:
        """Generate response using Mistral model"""
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(context_docs)
            
            # Create prompt for medical Q&A
            prompt = self._create_medical_prompt(query, context)
            
            # Call Hugging Face API
            response = self._call_mistral_api(prompt)
            
            # Post-process response
            answer = self._post_process_response(response)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, an error occurred while generating the response: {str(e)}"
    
    def _prepare_context(self, context_docs: List[Document]) -> str:
        """Prepare context from retrieved documents"""
        context = "\n".join([doc.page_content for doc in context_docs if doc.page_content])
        return context if context else "No relevant context found."
    
    def _create_medical_prompt(self, query: str, context: str) -> str:
        """Create a prompt for medical Q&A"""
        prompt = f"""[INST]You are a medical chatbot based on the Gale Encyclopedia of Medicine.

IMPORTANT GUIDELINES:
- Base your answer strictly on the provided context from the medical encyclopedia
- If the context doesn't contain relevant information, clearly state that
- Always include a disclaimer that this is for educational purposes only
- Recommend consulting healthcare professionals for medical advice
- Be precise and use medical terminology appropriately
- Do not speculate or provide information not supported by the context

CONTEXT FROM MEDICAL ENCYCLOPEDIA:
{context}

QUESTION: {query}

Please provide a comprehensive answer based on the medical encyclopedia context above. Include relevant medical details while maintaining clarity for general understanding.[/INST]"""
        
        return prompt
    
    def _call_mistral_api(self, prompt: str) -> str:
        """Call Mistral API via Hugging Face"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=120  # Increased from 60 to 120 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                else:
                    return str(result)
            else:
                error_msg = f"API request failed with status {response.status_code}"
                if response.text:
                    error_msg += f": {response.text}"
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            raise Exception("Request to AI model timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error occurred: {str(e)}")
        except Exception as e:
            raise Exception(f"Error calling AI model: {str(e)}")
    
    def _post_process_response(self, response: str) -> str:
        """Post-process the model response"""
        if not response:
            return "I couldn't generate a response. Please try rephrasing your question."
        
        # Clean up the response
        response = response.strip()
        
        # Add medical disclaimer if not present
        disclaimer = "\n\n**Medical Disclaimer:** This information is for educational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for medical concerns."
        
        if "disclaimer" not in response.lower() and "consult" not in response.lower():
            response += disclaimer
        
        return response
    
    def _extract_sources(self, docs: List[Document]) -> List[str]:
        """Extract source information from documents"""
        sources = []
        for doc in docs:
            if doc.metadata:
                source_info = f"Gale Encyclopedia of Medicine"
                if "chunk_id" in doc.metadata:
                    source_info += f" (Section {doc.metadata['chunk_id'] + 1})"
                sources.append(source_info)
            else:
                sources.append("Gale Encyclopedia of Medicine")
        
        return list(set(sources))  # Remove duplicates
    
    def get_medical_suggestions(self, query: str) -> List[str]:
        """Get medical topic suggestions based on query"""
        try:
            docs = self.vector_store.similarity_search(query, k=10)
            
            # Extract potential medical terms and topics
            suggestions = []
            for doc in docs:
                content = doc.page_content.lower()
                # Simple extraction of medical terms (this could be improved)
                words = content.split()
                for word in words:
                    if len(word) > 4 and word.isalpha():
                        suggestions.append(word.title())
            
            # Return unique suggestions
            return list(set(suggestions))[:5]
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {str(e)}")
            return []