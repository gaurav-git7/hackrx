from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Dict, Any
import requests
import tempfile
import os
from dotenv import load_dotenv
import re
import json
import mimetypes
import time
from io import BytesIO
import sys

# Load environment variables
load_dotenv()

# Add startup logging
print("🚀 Starting HackRx API...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Check if required environment variables are set
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Model priority for speed (fastest first) - FREE alternatives only
FAST_MODELS = ["mistral", "gemini", "huggingface"]

if not MISTRAL_API_KEY:
    print("⚠️ Warning: MISTRAL_API_KEY not found in environment variables")
if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not found in environment variables")
if not HF_TOKEN:
    print("⚠️ Warning: HF_TOKEN not found in environment variables")

# Determine which models are available (FREE alternatives only)
AVAILABLE_MODELS = []
if MISTRAL_API_KEY:
    AVAILABLE_MODELS.append("mistral")
if GEMINI_API_KEY:
    AVAILABLE_MODELS.append("gemini")
if HF_TOKEN:
    AVAILABLE_MODELS.append("huggingface")

print(f"🚀 Available models: {AVAILABLE_MODELS}")

# Test imports before creating the app
print("🔧 Testing imports...")
try:
    # Import our enhanced vector database functions
    from llm_query_retrieval_enhanced import (
        load_and_process_document,
        load_and_process_document_from_memory,
        create_semantic_chunks,
        create_vector_store,
        search_documents,
        answer_question,
        retrieve_relevant_chunks,
        is_confident,
        SimpleDocument,
        call_mistral_api,
        call_gemini_api,
        call_huggingface_api,
        clean_response
    )
    print("✅ Successfully imported all functions from llm_query_retrieval_enhanced")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Fallback imports
    try:
        from llm_query_retrieval_enhanced import (
            load_and_process_document,
            load_and_process_document_from_memory,
            answer_question
        )
        print("⚠️ Using fallback imports")
    except ImportError as e2:
        print(f"❌ Critical import error: {e2}")
        sys.exit(1)

print("✅ All imports successful")

# Create FastAPI app
app = FastAPI(title="HackRx Insurance Bot", version="1.0.0")
security = HTTPBearer()
AUTH_TOKEN = "02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"

# Simple request/response models (no Pydantic)
# We'll handle validation manually

def generate_fallback_answer(question, context):
    """Generate a simple answer from context without using external APIs"""
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

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the authentication token"""
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# 🔧 NEW: Confidence scoring function
def get_confidence_score(top_chunks):
    """Calculate confidence based on similarity scores with improved logic (tuple version)"""
    if not top_chunks:
        return 0.0
    best_score = min(chunk[1] for chunk in top_chunks)  # Use [1] for score
    confidence = max(0, 1 - (best_score * 2))  # Adjusted multiplier
    if len(top_chunks) >= 3:
        confidence = min(1.0, confidence + 0.1)
    if best_score < 0.5:
        confidence = min(1.0, confidence + 0.2)
    return confidence

def build_insurance_prompt(context, query):
    """Build a specific prompt for insurance policy questions"""
    return f"""You are an insurance policy expert. Based on the following policy document, provide a direct and concise answer to the question. Focus only on the key facts and avoid verbose explanations.

Policy Document:
{context}

Question: {query}

Provide a direct answer with specific details from the policy:"""

@app.on_event("startup")
async def startup_event():
    """Initialize app on startup"""
    print("🚀 Starting HackRx Document Q&A API...")
    print(f"📊 Available models: {AVAILABLE_MODELS}")
    print("✅ API is ready to serve requests!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx Insurance Bot API",
        "version": "1.0.0",
        "status": "running",
        "available_models": AVAILABLE_MODELS
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "Test endpoint working!",
        "timestamp": time.time(),
        "python_version": sys.version,
        "working_directory": os.getcwd()
    }

@app.post("/test-post")
async def test_post_endpoint(request: Request):
    """Test POST endpoint"""
    try:
        body = await request.json()
        return {
            "message": "POST test successful!",
            "received_data": body,
            "status": "success"
        }
    except Exception as e:
        return {
            "message": "POST test failed!",
            "error": str(e),
            "status": "error"
        }

# Removed old classes and functions - using our simplified implementation

@app.post("/hackrx/run")
async def hackrx_run(
    request: Request,
    token: str = Depends(verify_token)
):
    """
    Process document and answer questions using AI responses
    """
    try:
        print("🚀 Starting hackrx_run endpoint...")
        
        # Parse JSON request manually
        try:
            body = await request.json()
            print("✅ JSON parsed successfully")
        except Exception as e:
            print(f"❌ JSON parsing error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        documents_url = body.get("documents")
        questions = body.get("questions", [])
        
        if not documents_url:
            raise HTTPException(status_code=400, detail="Missing 'documents' field")
        if not questions:
            raise HTTPException(status_code=400, detail="Missing 'questions' field")
        if not isinstance(questions, list):
            raise HTTPException(status_code=400, detail="'questions' must be a list")
        
        print(f"🚀 Processing request with {len(questions)} questions")
        print(f"📥 Downloading document from: {documents_url}")
        
        # Step 1: Download document from URL
        try:
            response = requests.get(documents_url, timeout=30)
            response.raise_for_status()
            print("✅ Document downloaded successfully")
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=408, detail="Document download timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
        
        # Step 2: Process document in memory using BytesIO
        pdf_bytes = BytesIO(response.content)
        
        # Step 3: Load and process document directly from memory
        print("📄 Loading and processing document...")
        try:
            documents = load_and_process_document_from_memory(pdf_bytes, get_extension_from_url_or_header(documents_url, response))
            if not documents:
                raise HTTPException(status_code=500, detail="Failed to process document - no content extracted")
            print(f"✅ Processed {len(documents)} documents")
        except Exception as e:
            print(f"❌ Document processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        
        # Step 4: Create semantic chunks
        print("🔪 Creating semantic chunks...")
        try:
            chunks = create_semantic_chunks(documents, chunk_size=500, chunk_overlap=100)
            if not chunks:
                raise HTTPException(status_code=500, detail="Failed to create chunks")
            print(f"✅ Created {len(chunks)} chunks")
        except Exception as e:
            print(f"❌ Chunk creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create chunks: {str(e)}")
        
        # Step 5: Generate document store
        print("🤖 Creating document store...")
        try:
            vectorstore = create_vector_store(chunks)
            if not vectorstore:
                raise HTTPException(status_code=500, detail="Failed to create vector store")
            print(f"✅ Created vector store with {len(vectorstore)} items")
        except Exception as e:
            print(f"❌ Vector store creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create document store: {str(e)}")
        
        # Step 6: Process each question with real AI responses
        answers = []
        for i, question in enumerate(questions):
            try:
                print(f"🤔 Processing question {i+1}/{len(questions)}: {question[:50]}...")
                
                # Get relevant chunks
                try:
                    top_chunks = retrieve_relevant_chunks(question, vectorstore, top_k=3)
                    if not top_chunks:
                        print(f"⚠️ No relevant chunks found for question {i+1}")
                        answers.append("I couldn't find relevant information in the document to answer this question.")
                        continue
                except Exception as e:
                    print(f"❌ Chunk retrieval error for question {i+1}: {e}")
                    answers.append("Error retrieving relevant information from the document.")
                    continue
                
                # Convert chunks to the format expected by answer_question
                chunk_dicts = [{"chunk": chunk[0].page_content} for chunk in top_chunks]
                
                # Generate answer
                try:
                    answer = answer_question(question, chunk_dicts, method="auto")
                    if answer.startswith("Error:"):
                        print(f"⚠️ AI error for question {i+1}: {answer}")
                        # Try fallback
                        context = "\n\n".join([chunk[0].page_content for chunk in top_chunks])
                        answer = generate_fallback_answer(question, context)
                    
                    answers.append(answer)
                    print(f"✅ Answered question {i+1}")
                    
                except Exception as e:
                    print(f"❌ Answer generation error for question {i+1}: {e}")
                    # Use fallback answer
                    context = "\n\n".join([chunk[0].page_content for chunk in top_chunks])
                    answer = generate_fallback_answer(question, context)
                    answers.append(answer)
                
            except Exception as e:
                print(f"❌ Question processing error {i+1}: {e}")
                answers.append("An error occurred while processing this question.")
        
        # Step 7: Return results
        print("✅ Successfully processed all questions")
        return {
            "answers": answers,
            "status": "success",
            "questions_processed": len(questions),
            "chunks_created": len(chunks)
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Unexpected error in hackrx_run: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check environment variables
        env_status = {
            "MISTRAL_API_KEY": "✅" if MISTRAL_API_KEY else "❌",
            "GEMINI_API_KEY": "✅" if GEMINI_API_KEY else "❌", 
            "HF_TOKEN": "✅" if HF_TOKEN else "❌"
        }
        
        # Check available models
        model_status = {
            "mistral": "✅" if MISTRAL_API_KEY else "❌",
            "gemini": "✅" if GEMINI_API_KEY else "❌",
            "huggingface": "✅" if HF_TOKEN else "❌"
        }
        
        # Test basic imports
        import_status = "✅"
        try:
            from llm_query_retrieval_enhanced import answer_question
        except Exception as e:
            import_status = f"❌ {str(e)}"
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "environment_variables": env_status,
            "available_models": model_status,
            "imports": import_status,
            "available_models_count": len(AVAILABLE_MODELS)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

def get_extension_from_url_or_header(url, response):
    content_type = response.headers.get('Content-Type', '')
    if 'pdf' in content_type:
        return '.pdf'
    elif 'text' in content_type or 'plain' in content_type:
        return '.txt'
    # Fallback: guess from URL
    ext = os.path.splitext(url)[1]
    if ext in ['.pdf', '.txt']:
        return ext
    return '.bin'  # fallback

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)