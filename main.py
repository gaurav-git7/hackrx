from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Dict, Any
import requests
import tempfile
import os
from dotenv import load_dotenv
import re
import json
# Removed google.generativeai import - not needed for our implementation
# Removed unnecessary imports - using simplified implementation

# Load environment variables
load_dotenv()

# Check if required environment variables are set
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Model priority for speed (fastest first) - FREE alternatives only
FAST_MODELS = ["gemini", "huggingface"]

if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not found in environment variables")
if not HF_TOKEN:
    print("⚠️ Warning: HF_TOKEN not found in environment variables")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Determine which models are available (FREE alternatives only)
AVAILABLE_MODELS = []
if GEMINI_API_KEY:
    AVAILABLE_MODELS.append("gemini")
if HF_TOKEN:
    AVAILABLE_MODELS.append("huggingface")

print(f"🚀 Available FREE models: {AVAILABLE_MODELS}")

# Import our enhanced vector database functions
from llm_query_retrieval_enhanced import (
    load_and_process_document,
    create_semantic_chunks,
    create_vector_store,
    search_documents,
    answer_question,
    retrieve_relevant_chunks,
    is_confident
)

# Query expansion dictionary for insurance terms
INSURANCE_SYNONYMS = {
    "maternity": ["pregnancy", "childbirth", "delivery", "baby", "birth", "maternal"],
    "waiting period": ["exclusion period", "time limit", "coverage delay", "waiting time", "exclusion"],
    "pre-existing": ["pre-existing disease", "existing condition", "prior condition", "chronic"],
    "coverage": ["cover", "benefit", "protection", "insurance", "policy"],
    "exclusion": ["excluded", "not covered", "limitation", "restriction"],
    "premium": ["payment", "cost", "fee", "amount", "price"],
    "claim": ["claiming", "reimbursement", "payment", "benefit"],
    "hospital": ["hospitalization", "medical facility", "clinic", "healthcare center"],
    "surgery": ["operation", "procedure", "surgical", "medical procedure"],
    "medication": ["medicine", "drug", "prescription", "treatment"],
    "diagnosis": ["diagnostic", "test", "examination", "medical test"],
    "treatment": ["therapy", "care", "medical care", "healthcare"],
    "policy": ["insurance policy", "plan", "coverage", "contract"],
    "renewal": ["renew", "continue", "extend", "maintain"],
    "grace period": ["grace", "extension", "additional time", "late payment"]
}

def expand_query(query):
    """Expand query with insurance-related synonyms"""
    expanded_terms = []
    query_lower = query.lower()
    
    # Add original query
    expanded_terms.append(query)
    
    # Add synonyms for matched terms
    for term, synonyms in INSURANCE_SYNONYMS.items():
        if term in query_lower:
            for synonym in synonyms:
                expanded_query = query_lower.replace(term, synonym)
                if expanded_query != query_lower:
                    expanded_terms.append(expanded_query)
    
    # Add key terms from the query
    key_words = re.findall(r'\b\w+\b', query_lower)
    for word in key_words:
        if word in INSURANCE_SYNONYMS:
            for synonym in INSURANCE_SYNONYMS[word]:
                expanded_terms.append(synonym)
    
    # Remove duplicates and return unique queries
    unique_queries = list(set(expanded_terms))
    print(f"🔍 Expanded query '{query}' to {len(unique_queries)} variations")
    return unique_queries

def keyword_search(query, vectorstore, top_k=5):
    """Keyword-based search using exact word matching"""
    query_words = re.findall(r'\b\w+\b', query.lower())
    all_chunks = []
    
    # Search through document list
    for doc in vectorstore:
        doc_content = doc.page_content.lower()
        score = 0
        
        # Count exact word matches
        for word in query_words:
            if word in doc_content:
                score += 1
        
        # Normalize score by query length
        if len(query_words) > 0:
            score = score / len(query_words)
        
        if score > 0:
            all_chunks.append((doc, score))
    
    # Sort by score and return top_k
    all_chunks.sort(key=lambda x: x[1], reverse=True)
    return all_chunks[:top_k]

def hybrid_search(query, vectorstore, top_k=5):
    """Combine different search strategies"""
    # Get keyword results from retrieve_relevant_chunks
    semantic_results = retrieve_relevant_chunks(query, vectorstore, top_k=top_k)
    
    # Get additional keyword results
    keyword_results = keyword_search(query, vectorstore, top_k=top_k)
    
    # Combine and deduplicate
    combined = {}
    
    # Add semantic results
    for doc, score in semantic_results:
        combined[doc.page_content] = (doc, 1.0 - score)  # Convert to similarity score
    
    # Add keyword results
    for doc, score in keyword_results:
        if doc.page_content in combined:
            # Average the scores
            existing_score = combined[doc.page_content][1]
            combined[doc.page_content] = (doc, (existing_score + score) / 2)
        else:
            combined[doc.page_content] = (doc, score)
    
    # Sort by combined score and return top_k
    sorted_results = sorted(combined.values(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

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

# Initialize FastAPI app
app = FastAPI(title="HackRx Document Q&A API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize app on startup"""
    print("🚀 Starting HackRx Document Q&A API...")
    print(f"📊 Available models: {AVAILABLE_MODELS}")
    print("✅ API is ready to serve requests!")

# Security
security = HTTPBearer()
AUTH_TOKEN = "02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"

# Simple request/response models (no Pydantic)
# We'll handle validation manually

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
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
    return f"""
You are an expert insurance advisor. Answer the user's question using the information provided in the context below.

IMPORTANT: You MUST provide an answer. Do not say "insufficient information" or similar phrases.

If you find exact information, provide it clearly.
If you find related information, provide what you can find.
If you find partial information, provide it and mention what's missing.
If the context has any relevant information at all, use it to provide a helpful answer.

Context:
{context}

Question: {query}

Answer:
"""

@app.get("/")
async def root():
    return {
        "message": "HackRx Document Q&A API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "available_models": AVAILABLE_MODELS,
        "note": "Simplified version - document processing limited to text files"
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint that doesn't require document processing"""
    return {
        "message": "API is working!",
        "test": "success",
        "available_models": AVAILABLE_MODELS
    }

@app.post("/test-post")
async def test_post_endpoint(request: Request):
    """Test POST endpoint to check JSON parsing"""
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

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = None

    def process_document(self, file_path: str):
        try:
            loader = UnstructuredPDFLoader(file_path)
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)
            self.db = FAISS.from_documents(texts, self.embeddings)
            return True
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return False

    def get_relevant_context(self, query: str, k=3):
        if not self.db:
            return ""
        docs = self.db.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])

def download_pdf(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Could not download PDF")
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

@app.post("/hackrx/run")
async def process_query(request: QueryRequest):
    processor = DocumentProcessor()
    answers = []
    
    try:
        pdf_path = download_pdf(request.documents)
        if not processor.process_document(pdf_path):
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        for question in request.questions:
            context = processor.get_relevant_context(question)
            prompt = f"""Based on the following policy document content:
            {context}
            
            Question: {question}
            
            Provide a clear and concise answer based only on the information given above. 
            If the information is not found in the context, state that clearly."""

            response = model.generate_content(prompt)
            answers.append(response.text)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if 'pdf_path' in locals():
            try:
                os.unlink(pdf_path)
            except:
                pass
    
    return {"answers": answers}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "HackRx API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)