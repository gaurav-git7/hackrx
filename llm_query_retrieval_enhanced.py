import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Workaround for OpenMP runtime error on Windows (PyTorch + faiss)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Get HF_TOKEN but don't validate immediately - will check when needed
hf_token = os.environ.get("HF_TOKEN")

import json
import requests
from typing import List, Dict, Any

# LangChain imports for enhanced document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import PyPDF2

# 1. Enhanced Document Loading and Processing

def load_and_process_document(file_path: str) -> List[Any]:
    """Load and process document using LangChain with better semantic chunking"""
    if not file_path.endswith(".pdf"):
        raise ValueError("❌ Only PDF files are supported in this enhanced version.")
    
    print(f"📄 Loading document: {file_path}")
    
    try:
        # Try PyPDFLoader first (uses PyMuPDF under the hood)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"📝 Document loaded with PyPDFLoader: {len(documents)} pages")
    except Exception as e:
        print(f"⚠️ PyPDFLoader failed: {str(e)}")
        print("🔄 Falling back to PyPDF2...")
        
        # Fallback to PyPDF2
        documents = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    # Create a document object similar to LangChain's format
                    from langchain.schema import Document
                    doc = Document(
                        page_content=text,
                        metadata={"source": file_path, "page": page_num + 1}
                    )
                    documents.append(doc)
        
        print(f"📝 Document loaded with PyPDF2: {len(documents)} pages")
    
    return documents

def create_semantic_chunks(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    """Create semantic-aware chunks with improved parameters for insurance policies"""
    print("🔪 Creating semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} semantic chunks")
    return chunks

# 2. Enhanced Embedding and Vector Storage

def create_vector_store(chunks: List[Any], embedding_model_name: str = "huggingface/sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """Create FAISS vector store with enhanced embeddings"""
    print(f"🤖 Creating embeddings using {embedding_model_name}...")
    
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Force CPU to avoid GPU issues
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        print(f"✅ Vector store created with {len(chunks)} embeddings")
        return vectorstore
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        print("🔄 Trying with fallback embedding model...")
        # Fallback to a simpler model
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_model = HuggingFaceEmbeddings(
            model_name=fallback_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        print(f"✅ Vector store created with fallback model: {len(chunks)} embeddings")
        return vectorstore

def save_vector_store(vectorstore: FAISS, index_path: str = "faiss_index"):
    """Save FAISS index locally"""
    print(f"💾 Saving vector store to {index_path}...")
    vectorstore.save_local(index_path)
    print("✅ Vector store saved successfully")

def load_vector_store(index_path: str = "faiss_index", embedding_model_name: str = "huggingface/sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """Load existing FAISS index"""
    print(f"📂 Loading vector store from {index_path}...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded successfully")
        return vectorstore
    except Exception as e:
        print(f"❌ Error loading vector store: {str(e)}")
        raise

# 3. Enhanced Search and Retrieval

def search_documents(query: str, vectorstore: FAISS, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search documents using enhanced similarity search"""
    print(f"🔍 Searching for: '{query}'")
    
    # Use similarity search with metadata
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    
    results = []
    for i, (doc, score) in enumerate(docs_and_scores):
        results.append({
            "chunk": doc.page_content,
            "score": float(score),
            "index": i,
            "metadata": doc.metadata
        })
    
    print(f"✅ Found {len(results)} relevant chunks")
    return results

def is_confident(top_chunks, score_threshold=0.45):
    """Soft threshold for answer confidence based on FAISS similarity score (lower is better)"""
    if not top_chunks:
        return False
    # If using similarity_search_with_score, top_chunks is a list of (doc, score) tuples
    best_score = top_chunks[0][1] if isinstance(top_chunks[0], tuple) else top_chunks[0]["score"]
    return best_score < score_threshold

def retrieve_relevant_chunks(query, vectorstore, top_k=8):
    """Fetch the most semantically relevant chunks for a query, sorted by score (lowest first)"""
    # similarity_search_with_score returns list of (Document, score)
    top_chunks = vectorstore.similarity_search_with_score(query, k=top_k)
    # Sort by score (lowest = most similar)
    top_chunks = sorted(top_chunks, key=lambda x: x[1])
    return top_chunks

# 4. Enhanced Question Answering with External APIs

def answer_with_gemini(query: str, top_chunks: List[Dict[str, Any]], api_key: str = None, custom_prompt: str = None) -> str:
    """Use Google Gemini API for question answering"""
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Please set GEMINI_API_KEY environment variable or pass api_key parameter")
    context = "\n\n".join([c["chunk"] for c in top_chunks])
    prompt = custom_prompt if custom_prompt else f"You are a helpful assistant that answers questions based on the provided document context. \n\nInstructions:\n1. Answer the question based ONLY on the provided context\n2. If the answer is not in the context, say \"I cannot find the answer in the provided context\"\n3. Be specific and accurate - include exact numbers, dates, and conditions\n4. Use bullet points if the answer has multiple parts\n5. Keep answers concise but complete\n6. If the context mentions waiting periods, coverage limits, or specific conditions, include them\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 500,
            "temperature": 0.3
        }
    }
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}", 
            headers=headers, 
            json=data,
            timeout=30  # 30 second timeout
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except requests.exceptions.Timeout:
        return "Sorry, the request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Gemini API: {str(e)}"

def answer_with_huggingface(query: str, top_chunks: List[Dict[str, Any]], model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", custom_prompt: str = None) -> str:
    """Use Hugging Face Inference API for question answering"""
    if not hf_token:
        raise RuntimeError("Please set the HF_TOKEN environment variable")
    context = "\n\n".join([c["chunk"] for c in top_chunks])
    prompt = custom_prompt if custom_prompt else f"<s>[INST] Based on the following context, answer the question. If the answer cannot be found in the context, say \"I cannot find the answer in the provided context.\"\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.3,
            "do_sample": True
        }
    }
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model_name}", 
            headers=headers, 
            json=data,
            timeout=60  # 60 second timeout for HuggingFace
        )
        response.raise_for_status()
        return response.json()[0]["generated_text"].split("[/INST]")[-1].strip()
    except requests.exceptions.Timeout:
        return "Sorry, the request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with HuggingFace API: {str(e)}"

def answer_question(query: str, top_chunks: List[Dict[str, Any]], method: str = "gemini", custom_prompt: str = None, **kwargs) -> str:
    """Answer question using specified method"""
    if method.lower() == "gemini":
        return answer_with_gemini(query, top_chunks, custom_prompt=custom_prompt, **kwargs)
    elif method.lower() == "huggingface":
        return answer_with_huggingface(query, top_chunks, custom_prompt=custom_prompt, **kwargs)
    else:
        raise ValueError("method must be 'gemini' or 'huggingface'")

# 5. Enhanced Output Formatting

def format_output(query: str, answer: str, top_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "chunk": c["chunk"],
                "score": c["score"],
                "index": c["index"],
                "metadata": c.get("metadata", {})
            } for c in top_chunks
        ]
    }

# 6. Main Pipeline Functions

def create_document_index(file_path: str, index_path: str = "faiss_index") -> FAISS:
    """Complete pipeline to create document index"""
    print("🚀 Starting document indexing pipeline...")
    
    # Step 1: Load document
    documents = load_and_process_document(file_path)
    
    # Step 2: Create semantic chunks
    chunks = create_semantic_chunks(documents)
    
    # Step 3: Create vector store
    vectorstore = create_vector_store(chunks)
    
    # Step 4: Save vector store
    save_vector_store(vectorstore, index_path)
    
    print("✅ Document indexing completed!")
    return vectorstore

def query_documents(query: str, vectorstore: FAISS, method: str = "gemini") -> Dict[str, Any]:
    """Complete pipeline to query documents"""
    print(f"🔍 Processing query: '{query}'")
    
    # Step 1: Search for relevant chunks
    top_chunks = search_documents(query, vectorstore)
    
    # Step 2: Generate answer
    answer = answer_question(query, top_chunks, method=method)
    
    # Step 3: Format output
    output = format_output(query, answer, top_chunks)
    
    print("✅ Query processing completed!")
    return output

# Example usage
if __name__ == "__main__":
    # Configuration
    doc_path = "sample_policy.pdf"
    index_path = "faiss_index"
    
    # Check if index already exists
    if os.path.exists(index_path):
        print("📂 Loading existing index...")
        vectorstore = load_vector_store(index_path)
    else:
        print("🆕 Creating new index...")
        vectorstore = create_document_index(doc_path, index_path)
    
    # Example queries
    queries = [
        "What is the waiting period for pre-existing diseases?",
        "What are the coverage limits?",
        "What is the policy period?",
        "What are the exclusions?"
    ]
    
    for query in queries:
        print("\n" + "="*60)
        result = query_documents(query, vectorstore, method="gemini")
        print(json.dumps(result, indent=2, ensure_ascii=False)) 