import os
import re
import logging
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import PyPDF2
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF documents for the medical chatbot"""
    
    def __init__(self):
        """Initialize PDF processor"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process PDF and return document chunks
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Extract text from PDF
            text = self._extract_text_from_pdf(pdf_path)
            logger.info(f"Extracted {len(text)} characters from PDF")
            
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            logger.info(f"Cleaned text: {len(cleaned_text)} characters")
            
            # Split text into chunks
            documents = self._split_text_into_documents(cleaned_text, pdf_path)
            logger.info(f"Created {len(documents)} document chunks")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading PDF file: {str(e)}")
            raise
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'GALE ENCYCLOPEDIA OF MEDICINE.*?\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        
        # Remove table of contents patterns
        text = re.sub(r'\.{3,}', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', ' ', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _split_text_into_documents(self, text: str, source_path: str) -> List[Document]:
        """Split text into document chunks"""
        # Split text using the text splitter
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # Only include substantial chunks
                doc = Document(
                    page_content=chunk.strip(),
                    metadata={
                        "source": source_path,
                        "chunk_id": i,
                        "chunk_size": len(chunk)
                    }
                )
                documents.append(doc)
        
        return documents
    
    def extract_medical_terms(self, text: str) -> List[str]:
        """Extract potential medical terms from text"""
        # Simple medical term extraction using patterns
        medical_patterns = [
            r'\b[A-Z][a-z]*itis\b',  # Conditions ending in -itis
            r'\b[A-Z][a-z]*osis\b',  # Conditions ending in -osis  
            r'\b[A-Z][a-z]*emia\b',  # Conditions ending in -emia
            r'\b[A-Z][a-z]*pathy\b', # Conditions ending in -pathy
            r'\b[A-Z][a-z]*syndrome\b', # Syndromes
        ]
        
        terms = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text)
            terms.extend(matches)
        
        return list(set(terms))  # Remove duplicates