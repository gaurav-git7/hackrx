import os
import tempfile
import requests
from typing import List, Dict, Any, Tuple, Optional
import re
import pdfplumber
from PyPDF2 import PdfReader
import time
import random
from io import BytesIO
import numpy as np

# Try to import sentence-transformers, but make it optional
try:
    from sentence_transformers import SentenceTransformer
    # Initialize the embedding model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ all-MiniLM-L6-v2 embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        embedding_model = None
except ImportError:
    print("⚠️ sentence-transformers not available, using keyword search only")
    embedding_model = None

# Simple document class for our simplified version
class SimpleDocument:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
        # Generate embedding for the document
        if embedding_model:
            self.embedding = embedding_model.encode(page_content)
        else:
            self.embedding = None

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from a PDF or TXT file.
    - For .pdf: uses pdfplumber, then PyPDF2 as fallback.
    - For .txt: reads as UTF-8 text.
    - For other types: raises ValueError.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        # Try pdfplumber first
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"pdfplumber extraction failed: {str(e)}")
        # Fallback to PyPDF2
        try:
            text = ""
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"PyPDF2 extraction failed: {str(e)}")
        # If both fail, return error message
        return (
            "The PDF document could not be read properly. "
            "This might be due to: (1) The file being corrupted or password-protected, "
            "(2) The file being an image-based PDF (scanned document), "
            "(3) The file being in an unsupported format. "
            "Please provide a text-based PDF or convert the document to text format."
        )
    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading text file: {str(e)}"
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_and_process_document(file_path: str) -> list:
    """
    Load and process document from file path
    """
    try:
        text_content = extract_text_from_file(file_path)
        print(f"[DEBUG] Extracted text (first 500 chars): {text_content[:500]}")
        document = SimpleDocument(text_content, {"source": file_path})
        return [document]
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return []


def load_and_process_document_from_memory(file_bytes: BytesIO, file_extension: str) -> list:
    """
    Load and process document from BytesIO object in memory
    """
    try:
        print(f"[DEBUG] Processing file with extension: {file_extension}")
        print(f"[DEBUG] File bytes size: {len(file_bytes.getvalue())} bytes")
        
        if file_extension == ".pdf":
            print("[DEBUG] Processing as PDF...")
            # Process PDF directly from memory using pdfplumber
            try:
                text = ""
                print("[DEBUG] Opening with pdfplumber...")
                with pdfplumber.open(file_bytes) as pdf:
                    print(f"[DEBUG] PDF has {len(pdf.pages)} pages")
                    for i, page in enumerate(pdf.pages):
                        print(f"[DEBUG] Processing page {i+1}...")
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            print(f"[DEBUG] Page {i+1} extracted {len(page_text)} characters")
                        else:
                            print(f"[DEBUG] Page {i+1} extracted no text")
                
                if text.strip():
                    text_content = text.strip()
                    print(f"[DEBUG] Successfully extracted {len(text_content)} characters with pdfplumber")
                else:
                    print("[DEBUG] pdfplumber extracted no text, trying PyPDF2...")
                    # Fallback to PyPDF2
                    file_bytes.seek(0)  # Reset to beginning
                    reader = PdfReader(file_bytes)
                    text = ""
                    for i, page in enumerate(reader.pages):
                        print(f"[DEBUG] PyPDF2 processing page {i+1}...")
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            print(f"[DEBUG] PyPDF2 page {i+1} extracted {len(page_text)} characters")
                        else:
                            print(f"[DEBUG] PyPDF2 page {i+1} extracted no text")
                    text_content = text.strip() if text.strip() else "PDF could not be read properly"
                    print(f"[DEBUG] PyPDF2 extracted {len(text_content)} characters")
            except Exception as e:
                print(f"[ERROR] PDF extraction failed: {str(e)}")
                print(f"[ERROR] Exception type: {type(e).__name__}")
                import traceback
                print(f"[ERROR] Full traceback: {traceback.format_exc()}")
                text_content = "PDF could not be read properly"
        elif file_extension == ".txt":
            print("[DEBUG] Processing as text file...")
            # Read text from BytesIO
            file_bytes.seek(0)  # Reset to beginning
            try:
                text_content = file_bytes.read().decode('utf-8')
                print(f"[DEBUG] Successfully decoded {len(text_content)} characters as UTF-8")
            except UnicodeDecodeError as e:
                print(f"[DEBUG] UTF-8 decode failed: {e}, trying latin-1...")
                # Try other encodings
                file_bytes.seek(0)
                try:
                    text_content = file_bytes.read().decode('latin-1')
                    print(f"[DEBUG] Successfully decoded {len(text_content)} characters as latin-1")
                except Exception as e2:
                    print(f"[ERROR] All text decoding failed: {e2}")
                    text_content = "Text file could not be read properly"
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        print(f"[DEBUG] Final extracted text (first 500 chars): {text_content[:500]}")
        document = SimpleDocument(text_content, {"source": "memory"})
        print(f"[DEBUG] Created document with {len(text_content)} characters")
        return [document]
    except Exception as e:
        print(f"[ERROR] Error loading document from memory: {str(e)}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        return []


def create_semantic_chunks(documents: List[SimpleDocument], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[SimpleDocument]:
    """
    Create semantic chunks from documents with proper embeddings
    """
    chunks = []
    for doc in documents:
        text = doc.page_content
        if len(text) <= chunk_size:
            chunks.append(doc)
        else:
            # Split into overlapping chunks
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                chunk_doc = SimpleDocument(chunk_text, doc.metadata)
                chunks.append(chunk_doc)
                start = end - chunk_overlap
                if start >= len(text):
                    break
    return chunks


def create_vector_store(chunks: List[SimpleDocument]) -> List[SimpleDocument]:
    """
    Create a simple vector store (just return chunks with embeddings)
    """
    return chunks


def save_vector_store(vectorstore, file_path: str):
    """Save vector store (simplified - just for compatibility)"""
    pass


def load_vector_store(file_path: str):
    """Load vector store (simplified - just for compatibility)"""
    return []


def search_documents(query: str, vectorstore: List[SimpleDocument], top_k: int = 5) -> List[Tuple[SimpleDocument, float]]:
    """
    Search documents using semantic similarity with all-MiniLM-L6-v2 embeddings
    """
    if not embedding_model:
        # Fallback to keyword search
        return keyword_search(query, vectorstore, top_k)
    
    try:
        # Encode the query
        query_embedding = embedding_model.encode(query)
        
        # Calculate similarities
        similarities = []
        for doc in vectorstore:
            if doc.embedding is not None:
                similarity = np.dot(query_embedding, doc.embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding))
                similarities.append((doc, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    except Exception as e:
        print(f"Semantic search failed: {e}, falling back to keyword search")
        return keyword_search(query, vectorstore, top_k)


def keyword_search(query: str, vectorstore: List[SimpleDocument], top_k: int = 5) -> List[Tuple[SimpleDocument, float]]:
    """
    Simple keyword-based search as fallback
    """
    query_words = set(query.lower().split())
    results = []
    
    for doc in vectorstore:
        doc_words = set(doc.page_content.lower().split())
        intersection = query_words.intersection(doc_words)
        if intersection:
            score = len(intersection) / len(query_words)
            results.append((doc, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def retrieve_relevant_chunks(query: str, vectorstore: List[SimpleDocument], top_k: int = 5) -> List[Tuple[SimpleDocument, float]]:
    """Retrieve relevant chunks using semantic search"""
    return search_documents(query, vectorstore, top_k)


def call_mistral_api(prompt: str) -> str:
    """Call Mistral API with rate limiting and retry logic"""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return "Error: MISTRAL_API_KEY not found in environment variables"
    
    url = "https://api.mistral.ai/v1/chat/completions"
    
    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
            "temperature": 0.3
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Add random delay to avoid rate limits
    time.sleep(random.uniform(1, 3))
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "Error: No response from Mistral API"
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    return "Error: Mistral API rate limit exceeded. Please try again later."
            else:
                return f"Error communicating with Mistral API: {e}"
        except requests.exceptions.Timeout:
            return "Error: Mistral API request timed out"
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Mistral API: {e}"
        except Exception as e:
            return f"Unexpected error calling Mistral API: {str(e)}"
    
    return "Error: All retry attempts failed"


def call_gemini_api(prompt: str) -> str:
    """Call Gemini API with rate limiting and retry logic"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1000,
        }
    }
    
    # Add random delay to avoid rate limits
    time.sleep(random.uniform(1, 3))
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "Error: No response from Gemini API"
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    return "Error: Gemini API rate limit exceeded. Please try again later."
            else:
                return f"Error communicating with Gemini API: {e}"
        except requests.exceptions.Timeout:
            return "Error: Gemini API request timed out"
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Gemini API: {e}"
        except Exception as e:
            return f"Unexpected error calling Gemini API: {str(e)}"
    
    return "Error: All retry attempts failed"


def call_huggingface_api(prompt: str) -> str:
    """Call HuggingFace API with rate limiting and retry logic"""
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        return "Error: HF_TOKEN not found in environment variables"
    
    url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    data = {"inputs": prompt}
    
    # Add random delay to avoid rate limits
    time.sleep(random.uniform(1, 3))
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response from HuggingFace API")
            else:
                return "Error: No response from HuggingFace API"
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    return "Error: HuggingFace API rate limit exceeded. Please try again later."
            else:
                return f"Error communicating with HuggingFace API: {e}"
        except requests.exceptions.Timeout:
            return "Error: HuggingFace API request timed out"
        except requests.exceptions.RequestException as e:
            return f"Error communicating with HuggingFace API: {e}"
        except Exception as e:
            return f"Unexpected error calling HuggingFace API: {str(e)}"
    
    return "Error: All retry attempts failed"


def clean_response(response: str) -> str:
    """Clean and make the response more concise"""
    # Remove common verbose phrases
    verbose_phrases = [
        "Based on the provided context",
        "According to the policy document",
        "The context provided",
        "The policy states that",
        "Based on the information provided",
        "The document indicates that",
        "As mentioned in the policy",
        "The policy document shows that",
        "According to the provided information",
        "The context reveals that"
    ]
    
    cleaned = response
    for phrase in verbose_phrases:
        cleaned = cleaned.replace(phrase, "")
    
    # Remove excessive newlines and spaces
    cleaned = " ".join(cleaned.split())
    
    # Remove sentences that start with "To provide" or "For exact details"
    lines = cleaned.split(". ")
    filtered_lines = []
    for line in lines:
        if not any(line.startswith(phrase) for phrase in ["To provide", "For exact details", "For more information", "Please refer to", "You would need to"]):
            filtered_lines.append(line)
    
    cleaned = ". ".join(filtered_lines)
    
    return cleaned.strip()


def answer_question(question: str, top_chunks: List[Dict[str, str]], method: str = "auto", custom_prompt: str = None) -> str:
    """
    Answer question using AI with improved fallback handling
    """
    if not top_chunks:
        return "I couldn't find relevant information in the document to answer your question."
    
    # Build context from chunks
    context = "\n\n".join([chunk['chunk'] for chunk in top_chunks])
    
    # Use custom prompt if provided, otherwise build default
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = f"""Based on the following insurance policy document, provide a direct and concise answer to the question. Focus only on the key facts and avoid verbose explanations.

Document: {context}

Question: {question}

Answer (be direct and concise):"""
    
    # Try different methods based on availability and rate limits
    if method == "auto":
        # Try Mistral first, then Gemini, then HuggingFace, then fallback
        try:
            result = call_mistral_api(prompt)
            if not result.startswith("Error:"):
                return clean_response(result)
        except Exception as e:
            print(f"Mistral failed: {e}")
        
        # Fallback to Gemini
        try:
            result = call_gemini_api(prompt)
            if not result.startswith("Error:"):
                return clean_response(result)
        except Exception as e:
            print(f"Gemini failed: {e}")
        
        # Fallback to HuggingFace
        try:
            result = call_huggingface_api(prompt)
            if not result.startswith("Error:"):
                return clean_response(result)
        except Exception as e:
            print(f"HuggingFace failed: {e}")
        
        # Final fallback
        return generate_fallback_answer(question, context)
    
    elif method == "mistral":
        result = call_mistral_api(prompt)
        if result.startswith("Error:"):
            # If Mistral fails, try Gemini as backup
            backup_result = call_gemini_api(prompt)
            if not backup_result.startswith("Error:"):
                return clean_response(backup_result)
        return clean_response(result)
    
    elif method == "gemini":
        result = call_gemini_api(prompt)
        if result.startswith("Error:"):
            # If Gemini fails, try Mistral as backup
            backup_result = call_mistral_api(prompt)
            if not backup_result.startswith("Error:"):
                return clean_response(backup_result)
        return clean_response(result)
    
    elif method == "huggingface":
        result = call_huggingface_api(prompt)
        if result.startswith("Error:"):
            # If HuggingFace fails, try Mistral as backup
            backup_result = call_mistral_api(prompt)
            if not backup_result.startswith("Error:"):
                return clean_response(backup_result)
        return clean_response(result)
    
    else:
        return generate_fallback_answer(question, context)


def generate_fallback_answer(question: str, context: str) -> str:
    """Generate a fallback answer when all APIs fail"""
    return f"Based on the policy document, I found information that may be relevant to your question about {question}. Please review the document for specific details."


def is_confident(top_chunks: List[Tuple[SimpleDocument, float]], threshold: float = 0.7) -> bool:
    """Check if we have confident search results"""
    if not top_chunks:
        return False
    best_score = top_chunks[0][1] if top_chunks else 0
    return best_score >= threshold


def create_document_index(file_path: str, index_path: str = "faiss_index"):
    """Create document index (simplified - just for compatibility)"""
    pass


def query_documents(question: str, vectorstore: List[SimpleDocument], top_k: int = 5):
    """Query documents (simplified - just for compatibility)"""
    return search_documents(question, vectorstore, top_k) 