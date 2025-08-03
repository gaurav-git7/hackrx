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
    """Extract text from PDF using enhanced PyTorch-based extraction"""
    try:
        # Import the enhanced extractor
        from enhanced_pdf_extractor import pdf_extractor
        
        # Extract text using multiple strategies
        extracted_text = pdf_extractor.extract_text(pdf_path)
        
        # Clean the extracted text
        cleaned_text = pdf_extractor.clean_extracted_text(extracted_text)
        
        return cleaned_text
        
    except ImportError:
        # Fallback to basic extraction if enhanced extractor is not available
        print("Enhanced PDF extractor not available, using basic extraction")
        return _basic_extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"Enhanced PDF extraction failed: {str(e)}")
        return _basic_extract_text_from_pdf(pdf_path)

def _basic_extract_text_from_pdf(pdf_path: str) -> str:
    """Basic text extraction as fallback"""
    try:
        # Strategy 1: Try to read as text first (in case it's actually a text file)
        with open(pdf_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check if it looks like readable text
            if len(content) > 100 and not any(char in content for char in ['\x00', '\x01', '\x02', '\x03']):
                return content
    except:
        pass
    
    try:
        # Strategy 2: Read as binary and try multiple extraction methods
        with open(pdf_path, 'rb') as f:
            content = f.read()
        
        # Method 2a: Try UTF-8 decode with error handling
        try:
            decoded = content.decode('utf-8', errors='ignore')
            # Extract readable text lines
            lines = decoded.split('\n')
            text_content = ""
            for line in lines:
                # Remove binary artifacts and keep readable text
                clean_line = re.sub(r'[^\x20-\x7E\n\r\t]', '', line)
                if len(clean_line.strip()) > 5:  # Less restrictive
                    text_content += clean_line + '\n'
            
            if len(text_content.strip()) > 100:  # If we got substantial text
                return text_content.strip()
        except:
            pass
        
        # Method 2b: Look for PDF text objects (BT/ET markers)
        content_str = str(content)
        text_objects = re.findall(r'BT\s*(.*?)\s*ET', content_str, re.DOTALL)
        if text_objects:
            extracted_text = ""
            for obj in text_objects:
                # Extract text from text objects
                text_parts = re.findall(r'\(([^)]+)\)', obj)
                for part in text_parts:
                    if len(part.strip()) > 2:
                        extracted_text += part + ' '
            if extracted_text.strip():
                return extracted_text.strip()
        
        # Method 2c: Look for text patterns in binary data
        # Convert binary to string and look for readable patterns
        content_str = str(content)
        
        # Look for common insurance terms in the binary data
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
            # Try to extract sentences around found terms
            sentences = []
            for term in found_terms:
                # Find positions of the term
                term_positions = [m.start() for m in re.finditer(term, content_str, re.IGNORECASE)]
                for pos in term_positions:
                    # Extract text around the term
                    start = max(0, pos - 100)
                    end = min(len(content_str), pos + 100)
                    context = content_str[start:end]
                    # Clean the context
                    clean_context = re.sub(r'[^\x20-\x7E\n\r\t]', '', context)
                    if len(clean_context.strip()) > 20:
                        sentences.append(clean_context.strip())
            
            if sentences:
                return " ".join(sentences[:10])  # Return first 10 sentences
        
        # Method 2d: Try to extract any readable text patterns
        # Look for sequences of printable characters
        printable_pattern = re.findall(r'[A-Za-z0-9\s\.\,\;\:\!\?\-\(\)]+', content_str)
        if printable_pattern:
            # Filter out very short patterns and join
            meaningful_patterns = [p.strip() for p in printable_pattern if len(p.strip()) > 5]
            if meaningful_patterns:
                return " ".join(meaningful_patterns[:20])  # Return first 20 patterns
        
        # If all methods fail, return a helpful error message
        return "The PDF document could not be read properly. This might be due to:\n1. The file being corrupted or password-protected\n2. The file being an image-based PDF (scanned document)\n3. The file being in an unsupported format\n\nPlease provide a text-based PDF or convert the document to text format."
        
    except Exception as e:
        return f"Error processing the PDF file: {str(e)}. Please ensure the file is accessible and in a supported format."

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
    
    # Check if context contains error messages or is too short
    if "error" in context_lower or "could not" in context_lower or "failed" in context_lower:
        return f"I apologize, but I'm unable to read the policy document properly. The document appears to be in a format that cannot be processed. To answer your question about '{question}', please provide the policy document in a text format or ensure it's a readable PDF file."
    
    if len(context.strip()) < 100:
        return f"I apologize, but the policy document contains very little readable text. To answer your question about '{question}', please provide a properly formatted policy document with clear text content."
    
    # Look for key terms in the question
    key_terms = []
    for term in ["maternity", "waiting period", "pre-existing", "coverage", "exclusion", "premium", "claim", "hospital", "surgery", "medication", "diagnosis", "treatment", "policy", "renewal", "grace period", "ayush", "cataract", "organ donor", "ncd", "no claim discount", "preventive", "health check"]:
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
        # If no specific terms found, provide a helpful response
        return f"I've reviewed the policy document, but I couldn't find specific information about '{question}'. This could mean either:\n1. The information is not covered in this policy\n2. The document format prevents proper text extraction\n3. The terms might be described differently in the policy\n\nPlease provide a properly formatted text version of the policy document for more accurate answers."

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