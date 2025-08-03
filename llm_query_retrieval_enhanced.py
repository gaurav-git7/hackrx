import os
import tempfile
import requests
from typing import List, Dict, Any, Tuple, Optional
import re

# Simple document class for our simplified version
class SimpleDocument:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using basic text extraction"""
    try:
        # Try to read as text first (in case it's actually a text file)
        with open(pdf_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check if it looks like readable text
            if len(content) > 100 and not any(char in content for char in ['\x00', '\x01', '\x02', '\x03']):
                return content
    except:
        pass
    
    try:
        # Try to read as binary and extract text
        with open(pdf_path, 'rb') as f:
            content = f.read()
            
        # Simple text extraction from binary content
        # Look for text patterns in the binary data
        text_content = ""
        
        # Try to decode as UTF-8
        try:
            decoded = content.decode('utf-8', errors='ignore')
            # Extract readable text
            lines = decoded.split('\n')
            for line in lines:
                # Check if line contains readable text
                if len(line.strip()) > 10 and not any(char in line for char in ['\x00', '\x01', '\x02', '\x03']):
                    # Remove binary artifacts
                    clean_line = re.sub(r'[^\x20-\x7E\n\r\t]', '', line)
                    if len(clean_line.strip()) > 5:
                        text_content += clean_line + '\n'
        except:
            pass
        
        # If no readable text found, try to extract from PDF structure
        if not text_content.strip():
            # Look for PDF text objects
            content_str = str(content)
            # Find text between BT and ET markers (PDF text objects)
            text_objects = re.findall(r'BT\s*(.*?)\s*ET', content_str, re.DOTALL)
            for obj in text_objects:
                # Extract text from text objects
                text_parts = re.findall(r'\(([^)]+)\)', obj)
                for part in text_parts:
                    if len(part.strip()) > 2:
                        text_content += part + ' '
        
        return text_content.strip() if text_content.strip() else "PDF content could not be extracted. Please provide a text file instead."
        
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}. Please provide a text file instead."

def load_and_process_document(file_path: str) -> List[SimpleDocument]:
    """
    Load and process document from file path
    """
    try:
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # Extract text from PDF
            text_content = extract_text_from_pdf(file_path)
        else:
            # Read as text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        
        # Create a single document
        document = SimpleDocument(text_content, {"source": file_path})
        return [document]
        
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return [SimpleDocument(f"Error loading document: {str(e)}", {"source": file_path})]

def create_semantic_chunks(documents: List[SimpleDocument], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[SimpleDocument]:
    """
    Create semantic chunks from documents
    """
    chunks = []
    
    for doc in documents:
        text = doc.page_content
        words = text.split()
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunk_doc = SimpleDocument(
                    chunk_text,
                    {**doc.metadata, "chunk_index": i // (chunk_size - chunk_overlap)}
                )
                chunks.append(chunk_doc)
    
    return chunks

def create_vector_store(chunks: List[SimpleDocument]) -> List[SimpleDocument]:
    """
    Create a simple document store (no vector embeddings in simplified version)
    """
    return chunks

def save_vector_store(vectorstore, file_path: str):
    """
    Save vector store (not implemented in simplified version)
    """
    print("Vector store saving not implemented in simplified version")

def load_vector_store(file_path: str):
    """
    Load vector store (not implemented in simplified version)
    """
    print("Vector store loading not implemented in simplified version")
    return []

def search_documents(query: str, vectorstore: List[SimpleDocument], top_k: int = 5) -> List[Tuple[SimpleDocument, float]]:
    """
    Search documents using keyword matching
    """
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    results = []
    
    for doc in vectorstore:
        doc_words = set(re.findall(r'\b\w+\b', doc.page_content.lower()))
        
        # Calculate similarity score
        if query_words:
            intersection = query_words.intersection(doc_words)
            score = len(intersection) / len(query_words)
        else:
            score = 0
        
        if score > 0:
            results.append((doc, score))
    
    # Sort by score and return top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def retrieve_relevant_chunks(query: str, vectorstore: List[SimpleDocument], top_k: int = 5) -> List[Tuple[SimpleDocument, float]]:
    """
    Retrieve relevant chunks using keyword search
    """
    print("🔍 Using keyword-based search")
    return search_documents(query, vectorstore, top_k)

def answer_question(question: str, top_chunks: List[Dict[str, str]], method: str = "gemini", custom_prompt: str = None) -> str:
    """
    Answer question using AI models with improved fallback handling
    """
    # Prepare context from chunks
    context = "\n\n".join([chunk['chunk'] for chunk in top_chunks])
    
    # Use custom prompt if provided, otherwise use default
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = f"""
        Based on the following context, answer the question. If the context doesn't contain enough information, say so clearly.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
    
    # Try Gemini first
    if method == "gemini" or method == "auto":
        try:
            return call_gemini_api(prompt)
        except Exception as e:
            print(f"❌ Gemini failed: {str(e)}")
            if method == "auto":
                # Fallback to HuggingFace
                try:
                    return call_huggingface_api(prompt)
                except Exception as e2:
                    print(f"❌ HuggingFace also failed: {str(e2)}")
                    return generate_fallback_answer(question, context)
            else:
                raise e
    
    # Try HuggingFace
    elif method == "huggingface":
        try:
            return call_huggingface_api(prompt)
        except Exception as e:
            print(f"❌ HuggingFace failed: {str(e)}")
            return generate_fallback_answer(question, context)
    
    else:
        return generate_fallback_answer(question, context)

def call_gemini_api(prompt: str) -> str:
    """
    Call Gemini API with improved error handling
    """
    import os
    import requests
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not found in environment variables")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
        
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise Exception("No response from Gemini API")
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            raise Exception(f"Rate limit exceeded for Gemini API. Please wait a moment and try again.")
        else:
            raise Exception(f"Gemini API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"Error communicating with Gemini API: {str(e)}")

def call_huggingface_api(prompt: str) -> str:
    """
    Call HuggingFace API with improved error handling
    """
    import os
    import requests
    
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise Exception("HF_TOKEN not found in environment variables")
    
    url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    data = {"inputs": prompt}
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0]["generated_text"]
        else:
            raise Exception("No response from HuggingFace API")
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            raise Exception(f"Rate limit exceeded for HuggingFace API. Please wait a moment and try again.")
    else:
            raise Exception(f"HuggingFace API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"Error communicating with HuggingFace API: {str(e)}")

def generate_fallback_answer(question: str, context: str) -> str:
    """
    Generate a simple answer from context without using external APIs
    """
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Look for key terms in the question
    key_terms = []
    for term in ["maternity", "waiting period", "pre-existing", "coverage", "exclusion", "premium", "claim", "hospital", "surgery", "medication", "diagnosis", "treatment", "policy", "renewal", "grace period"]:
        if term in question_lower:
            key_terms.append(term)
    
    # Find relevant sentences from context
    sentences = context.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for term in key_terms:
            if term in sentence_lower:
                relevant_sentences.append(sentence.strip())
                break
    
    if relevant_sentences:
        # Return the most relevant sentence
        return relevant_sentences[0] + "."
    else:
        # If no specific terms found, return a general response
        return f"Based on the policy document, I found information that may be relevant to your question about {question}. Please review the document for specific details."

def is_confident(top_chunks: List[Tuple[SimpleDocument, float]], threshold: float = 0.7) -> bool:
    """
    Check if we have enough confidence in the search results
    """
    if not top_chunks:
        return False
    
    # Calculate average score
    scores = [score for _, score in top_chunks]
    avg_score = sum(scores) / len(scores)
    
    return avg_score >= threshold

def create_document_index(file_path: str, index_path: str = "faiss_index"):
    """
    Create document index (simplified version)
    """
    documents = load_and_process_document(file_path)
    chunks = create_semantic_chunks(documents)
    vectorstore = create_vector_store(chunks)
    return vectorstore

def query_documents(question: str, vectorstore: List[SimpleDocument], top_k: int = 5):
    """
    Query documents and get answer
    """
    chunks = retrieve_relevant_chunks(question, vectorstore, top_k)
    formatted_chunks = [{'chunk': chunk[0].page_content} for chunk in chunks]
    return answer_question(question, formatted_chunks, method="auto") 