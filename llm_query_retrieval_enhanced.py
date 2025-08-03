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
import concurrent.futures
import threading
import hashlib
import pickle

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

# Global session for connection pooling
session = requests.Session()
session.headers.update({
    'User-Agent': 'InsuranceBot/1.0'
})

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

# Simple cache for processed documents
document_cache = {}

def get_document_hash(file_bytes: BytesIO) -> str:
    """Generate hash for document caching"""
    file_bytes.seek(0)
    content = file_bytes.read()
    return hashlib.md5(content).hexdigest()

def load_cached_document(file_hash: str) -> Optional[List[SimpleDocument]]:
    """Load cached document if available"""
    cache_file = f"cache_{file_hash}.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load cache: {e}")
    return None

def save_cached_document(file_hash: str, documents: List[SimpleDocument]):
    """Save processed documents to cache"""
    cache_file = f"cache_{file_hash}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(documents, f)
    except Exception as e:
        print(f"Failed to save cache: {e}")


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
    Load and process document from memory with caching
    """
    try:
        # Generate hash for caching
        file_hash = get_document_hash(file_bytes)
        
        # Try to load from cache first
        cached_docs = load_cached_document(file_hash)
        if cached_docs:
            print("✅ Loaded from cache")
            return cached_docs
        
        # Process document if not cached
        print("🔄 Processing document...")
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            file_bytes.seek(0)
            temp_file.write(file_bytes.read())
            temp_file_path = temp_file.name
        
        try:
            # Extract text
            text_content = extract_text_from_file(temp_file_path)
            print(f"[DEBUG] Extracted text length: {len(text_content)}")
            
            # Create document
            document = SimpleDocument(text_content, {"source": "uploaded_file"})
            
            # Create chunks
            chunks = create_semantic_chunks([document])
            print(f"[DEBUG] Created {len(chunks)} chunks")
            
            # Cache the processed documents
            save_cached_document(file_hash, chunks)
            
            return chunks
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file: {e}")
                
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return []


def create_semantic_chunks(documents: List[SimpleDocument], chunk_size: int = 500, chunk_overlap: int = 100) -> List[SimpleDocument]:
    """
    Create smaller, more focused chunks for faster processing
    """
    chunks = []
    
    for doc in documents:
        text = doc.page_content
        if len(text) <= chunk_size:
            chunks.append(doc)
            continue
            
        # Split into smaller chunks
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
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


def search_documents(query: str, vectorstore: List[SimpleDocument], top_k: int = 3) -> List[Tuple[SimpleDocument, float]]:
    """
    Optimized semantic search with reduced results
    """
    if not embedding_model:
        return keyword_search(query, vectorstore, top_k)
    
    try:
        query_embedding = embedding_model.encode(query)
        
        results = []
        for doc in vectorstore:
            if doc.embedding is not None:
                similarity = np.dot(query_embedding, doc.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
                )
                results.append((doc, similarity))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
        
    except Exception as e:
        print(f"Semantic search failed: {e}")
        return keyword_search(query, vectorstore, top_k)


def keyword_search(query: str, vectorstore: List[SimpleDocument], top_k: int = 3) -> List[Tuple[SimpleDocument, float]]:
    """
    Optimized keyword search with reduced results
    """
    query_words = set(query.lower().split())
    
    results = []
    for doc in vectorstore:
        doc_words = set(doc.page_content.lower().split())
        intersection = query_words.intersection(doc_words)
        if intersection:
            score = len(intersection) / len(query_words)
            results.append((doc, score))
    
    # Sort by score and return top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def retrieve_relevant_chunks(query: str, vectorstore: List[SimpleDocument], top_k: int = 3) -> List[Tuple[SimpleDocument, float]]:
    """
    Optimized chunk retrieval with reduced results
    """
    return search_documents(query, vectorstore, top_k)


def call_mistral_api(prompt: str) -> str:
    """Call Mistral API with optimized performance"""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return "Error: MISTRAL_API_KEY not found in environment variables"
    
    url = "https://api.mistral.ai/v1/chat/completions"
    
    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,  # Reduced for faster response
        "temperature": 0.3
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    max_retries = 2  # Reduced retries
    for attempt in range(max_retries):
        try:
            response = session.post(url, json=data, headers=headers, timeout=15)  # Reduced timeout
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "Error: No response from Mistral API"
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = min(1 + attempt, 3)  # Shorter backoff
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
    """Call Gemini API with optimized performance"""
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
            "maxOutputTokens": 500,  # Reduced for faster response
        }
    }
    
    max_retries = 2  # Reduced retries
    for attempt in range(max_retries):
        try:
            response = session.post(url, json=data, timeout=15)  # Reduced timeout
            response.raise_for_status()
            
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "Error: No response from Gemini API"
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = min(1 + attempt, 3)  # Shorter backoff
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
    """Call HuggingFace API with optimized performance"""
    api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        return "Error: HUGGINGFACE_API_KEY not found in environment variables"
    
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "inputs": f"<s>[INST] {prompt} [/INST]",
        "parameters": {
            "max_new_tokens": 300,  # Reduced for faster response
            "temperature": 0.3,
            "do_sample": True
        }
    }
    
    max_retries = 2  # Reduced retries
    for attempt in range(max_retries):
        try:
            response = session.post(url, json=data, headers=headers, timeout=15)  # Reduced timeout
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]["generated_text"]
            else:
                return "Error: No response from HuggingFace API"
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = min(1 + attempt, 3)  # Shorter backoff
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


def call_apis_parallel(prompt: str) -> str:
    """Call multiple APIs in parallel and return the first successful response"""
    
    def call_api(api_name: str, api_func):
        try:
            result = api_func(prompt)
            if not result.startswith("Error:"):
                return result
            return None
        except Exception as e:
            print(f"{api_name} failed: {e}")
            return None
    
    # Define available APIs
    apis = [
        ("Mistral", call_mistral_api),
        ("Gemini", call_gemini_api),
        ("HuggingFace", call_huggingface_api)
    ]
    
    # Try APIs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_api = {
            executor.submit(call_api, name, func): name 
            for name, func in apis
        }
        
        for future in concurrent.futures.as_completed(future_to_api, timeout=20):
            api_name = future_to_api[future]
            try:
                result = future.result()
                if result:
                    print(f"✅ {api_name} responded successfully")
                    return result
            except Exception as e:
                print(f"❌ {api_name} failed: {e}")
    
    # If all parallel calls fail, try sequential fallback
    print("🔄 Trying sequential fallback...")
    for name, func in apis:
        try:
            result = func(prompt)
            if not result.startswith("Error:"):
                return result
        except Exception as e:
            print(f"❌ {name} fallback failed: {e}")
    
    return "Error: All API calls failed"


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


def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"⏱️ {func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

@monitor_performance
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
        # Try parallel API calls first
        result = call_apis_parallel(prompt)
        if not result.startswith("Error:"):
            return clean_response(result)
        
        # Fallback to sequential
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