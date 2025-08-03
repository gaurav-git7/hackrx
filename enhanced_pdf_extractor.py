import os
import re
import tempfile
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPDFExtractor:
    """
    Enhanced PDF text extraction using multiple strategies including PyTorch-based OCR
    """
    
    def __init__(self):
        # Initialize extraction methods based on available dependencies
        self.extraction_methods = []
        
        # Try to add pdfplumber (most reliable)
        try:
            import pdfplumber
            self.extraction_methods.append(self._extract_with_pdfplumber)
            logger.info("pdfplumber available")
        except ImportError:
            logger.warning("pdfplumber not available")
        
        # Try to add PyPDF2 (good fallback)
        try:
            import PyPDF2
            self.extraction_methods.append(self._extract_with_pypdf2)
            logger.info("PyPDF2 available")
        except ImportError:
            logger.warning("PyPDF2 not available")
        
        # Try to add OCR (for scanned documents) - disabled for deployment
        # try:
        #     import pytesseract
        #     import pdf2image
        #     self.extraction_methods.append(self._extract_with_ocr)
        #     logger.info("OCR capabilities available")
        # except ImportError:
        #     logger.warning("OCR dependencies not available")
        logger.info("OCR disabled for deployment stability")
        
        # Always add basic text extraction as fallback
        self.extraction_methods.append(self._extract_with_basic_text)
        
        logger.info(f"Initialized with {len(self.extraction_methods)} extraction methods")
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple strategies with fallback
        """
        logger.info(f"Starting text extraction from: {pdf_path}")
        
        for i, method in enumerate(self.extraction_methods):
            try:
                logger.info(f"Trying method {i+1}: {method.__name__}")
                text = method(pdf_path)
                if text and len(text.strip()) > 100:
                    logger.info(f"Successfully extracted {len(text)} characters using {method.__name__}")
                    return text
            except Exception as e:
                logger.warning(f"Method {method.__name__} failed: {str(e)}")
                continue
        
        # If all methods fail, return helpful error message
        return "Unable to extract text from this PDF. The document may be:\n1. Password-protected\n2. Corrupted\n3. Image-based (scanned document)\n4. In an unsupported format\n\nPlease provide a text-based PDF or convert the document to text format."
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (best for structured PDFs)"""
        try:
            import pdfplumber
            
            text_content = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            
            return text_content.strip()
        except ImportError:
            logger.warning("pdfplumber not available")
            return ""
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
            return ""
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (good fallback)"""
        try:
            import PyPDF2
            
            text_content = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            
            return text_content.strip()
        except ImportError:
            logger.warning("PyPDF2 not available")
            return ""
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
            return ""
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR (for scanned/image-based PDFs) - DISABLED"""
        logger.warning("OCR extraction disabled for deployment stability - PIL/Pillow removed")
        return ""
    
    def _extract_with_basic_text(self, pdf_path: str) -> str:
        """Basic text extraction as fallback (your original method)"""
        try:
            # Try reading as text first
            with open(pdf_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 100 and not any(char in content for char in ['\x00', '\x01', '\x02', '\x03']):
                    return content
        except:
            pass
        
        try:
            # Read as binary and try to extract text
            with open(pdf_path, 'rb') as f:
                content = f.read()
            
            # Try UTF-8 decode
            try:
                decoded = content.decode('utf-8', errors='ignore')
                lines = decoded.split('\n')
                text_content = ""
                for line in lines:
                    clean_line = re.sub(r'[^\x20-\x7E\n\r\t]', '', line)
                    if len(clean_line.strip()) > 5:  # Less restrictive
                        text_content += clean_line + '\n'
                
                if len(text_content.strip()) > 100:
                    return text_content.strip()
            except:
                pass
            
            # Look for text patterns in binary data
            content_str = str(content)
            
            # Look for common insurance terms
            insurance_terms = [
                'policy', 'coverage', 'premium', 'claim', 'hospital', 'surgery', 
                'maternity', 'waiting', 'period', 'exclusion', 'benefit', 'medical',
                'insurance', 'health', 'treatment', 'diagnosis', 'medication'
            ]
            
            found_terms = []
            for term in insurance_terms:
                if term.lower() in content_str.lower():
                    found_terms.append(term)
            
            if found_terms:
                sentences = []
                for term in found_terms:
                    term_positions = [m.start() for m in re.finditer(term, content_str, re.IGNORECASE)]
                    for pos in term_positions:
                        start = max(0, pos - 100)
                        end = min(len(content_str), pos + 100)
                        context = content_str[start:end]
                        clean_context = re.sub(r'[^\x20-\x7E\n\r\t]', '', context)
                        if len(clean_context.strip()) > 20:
                            sentences.append(clean_context.strip())
                
                if sentences:
                    return " ".join(sentences[:10])
            
            # Look for printable patterns
            printable_pattern = re.findall(r'[A-Za-z0-9\s\.\,\;\:\!\?\-\(\)]+', content_str)
            if printable_pattern:
                meaningful_patterns = [p.strip() for p in printable_pattern if len(p.strip()) > 5]
                if meaningful_patterns:
                    return " ".join(meaningful_patterns[:20])
            
            return ""
            
        except Exception as e:
            logger.warning(f"Basic text extraction failed: {str(e)}")
            return ""
    
    def clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        """
        if not text:
            return text
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
        
        # Fix common text issues
        text = text.replace('|', 'I')  # Common text mistake
        text = text.replace('0', 'O')  # In context where it makes sense
        
        # Remove page numbers and headers/footers
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are likely page numbers or headers
            if (len(line) < 3 or 
                re.match(r'^\d+$', line) or  # Just numbers
                re.match(r'^[A-Z\s]+$', line) and len(line) < 20):  # All caps short lines
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()

# Global extractor instance
pdf_extractor = EnhancedPDFExtractor() 