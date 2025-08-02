from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import requests
import tempfile
import os
from dotenv import load_dotenv
import re

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
    
    # Get all documents from vectorstore
    all_docs = vectorstore.docstore._dict
    
    for doc_id, doc in all_docs.items():
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
    """Combine semantic and keyword search"""
    # Get semantic results
    semantic_results = retrieve_relevant_chunks(query, vectorstore, top_k=top_k)
    
    # Get keyword results
    keyword_results = keyword_search(query, vectorstore, top_k=top_k)
    
    # Combine and deduplicate
    combined = {}
    
    # Add semantic results (FAISS scores are distance-based, so lower is better)
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

# Security
security = HTTPBearer()
AUTH_TOKEN = "02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"

# Pydantic models
class HackRxRequest(BaseModel):
    documents: str  # URL to PDF file
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

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
    return {"message": "HackRx Document Q&A API is running!"}

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """
    Process PDF document and answer questions using vector database with real AI responses
    """
    try:
        print(f"�� Processing request with {len(request.questions)} questions")
        
        # Step 1: Download PDF from URL
        print(f"📥 Downloading PDF from: {request.documents}")
        response = requests.get(request.documents)
        response.raise_for_status()
        
        # Step 2: Save PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            pdf_path = tmp_file.name
        
        try:
            # Step 3: Load and process document
            print("📄 Loading and processing document...")
            # SUGGESTED: Try chunk_size=1200-2000, chunk_overlap=200-400 for insurance policies
            documents = load_and_process_document(pdf_path)
            
            # Step 4: Create semantic chunks
            print("🔪 Creating semantic chunks...")
            chunks = create_semantic_chunks(documents)  # Now uses improved chunking
            
            # Step 5: Generate embeddings and build FAISS index
            print("🤖 Creating vector store...")
            vectorstore = create_vector_store(chunks)
            
            # Step 6: Process each question with real AI responses
            answers = []
            for i, question in enumerate(request.questions, 1):
                print(f"\n---\n🔍 Query {i}: {question}")
                print(f"🔍 Processing question {i}: {question}")
                
                # 🔧 NEW: Query expansion
                expanded_queries = expand_query(question)
                
                # 🔧 NEW: Multi-strategy search
                all_chunks = []
                
                # Strategy 1: Original semantic search
                print("🔍 Strategy 1: Semantic search")
                semantic_chunks = retrieve_relevant_chunks(question, vectorstore, top_k=5)
                all_chunks.extend(semantic_chunks)
                
                # Strategy 2: Hybrid search
                print("🔍 Strategy 2: Hybrid search")
                hybrid_chunks = hybrid_search(question, vectorstore, top_k=5)
                all_chunks.extend(hybrid_chunks)
                
                # Strategy 3: Expanded query search
                print("🔍 Strategy 3: Expanded query search")
                for expanded_query in expanded_queries[:5]:  # Use top 5 expanded queries
                    expanded_chunks = retrieve_relevant_chunks(expanded_query, vectorstore, top_k=3)
                    all_chunks.extend(expanded_chunks)
                
                # Deduplicate and get top chunks
                unique_chunks = {}
                for doc, score in all_chunks:
                    if doc.page_content not in unique_chunks:
                        unique_chunks[doc.page_content] = (doc, score)
                
                # Sort by score and get top 8 (increased from 5)
                sorted_chunks = sorted(unique_chunks.values(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                top_chunks = sorted_chunks[:8]
                
                # Print FAISS scores for debugging
                print("Top FAISS scores:", [round(c[1], 4) for c in top_chunks])
                best_score = top_chunks[0][1] if top_chunks else None
                print(f"Best FAISS score: {best_score}")
                
                #  IMPROVED: Use real AI to generate answer with fallback handling
                context = "\n\n".join([c[0].page_content for c in top_chunks])
                print(f"Context passed to LLM (first 500 chars):\n{context[:500]}\n---")
                prompt = build_insurance_prompt(context, question)
                
                # 🔧 NEW: Check confidence and generate answer
                try:
                    # Convert tuples to expected dictionary format
                    formatted_chunks = [{'chunk': c[0].page_content} for c in top_chunks]
                    print(f"🔧 Calling Gemini with {len(formatted_chunks)} chunks")
                    answer = answer_question(
                        question,
                        top_chunks=formatted_chunks,
                        method="gemini",
                        custom_prompt=prompt
                    )
                    print("LLM (Gemini) answer:", answer)
                except Exception as e:
                    print(f"❌ Gemini failed with error: {str(e)}")
                    print(f"❌ Error type: {type(e).__name__}")
                    try:
                        # Convert tuples to expected dictionary format
                        formatted_chunks = [{'chunk': c[0].page_content} for c in top_chunks]
                        print(f"🔧 Calling HuggingFace with {len(formatted_chunks)} chunks")
                        answer = answer_question(
                            question,
                            top_chunks=formatted_chunks,
                            method="huggingface",
                            custom_prompt=prompt
                        )
                        print("LLM (HuggingFace) answer:", answer)
                    except Exception as e2:
                        print(f"❌ HuggingFace failed with error: {str(e2)}")
                        print(f"❌ Error type: {type(e2).__name__}")
                        answer = "Unable to process this question at the moment."
                
                answers.append(answer.strip())
                print(f"✅ Final answer for query {i}: {answer.strip()}")
            
            print(f"🎉 Successfully processed {len(answers)} questions")
            return HackRxResponse(answers=answers)
            
        finally:
            # Clean up temporary file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
                
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "HackRx API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)