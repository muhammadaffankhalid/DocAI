from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from llama_cpp import Llama
import os
import PyPDF2
import fitz  # PyMuPDF
from io import BytesIO
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware 
# Initialize FastAPI
app = FastAPI()
# Enable CORS for Next.js frontend
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Thread pool for model inference
executor = ThreadPoolExecutor(max_workers=1)

# Load Mistral model with speed optimizations
try:
    llm = Llama(
        model_path="/Users/affankhalid/Desktop/docai/back-end/models/mistral-7b-instruct-v0.1.Q2_K.gguf",
        n_ctx=1024,  # Smaller context for speed
        n_threads=os.cpu_count(),  # Use all CPU cores
        n_batch=8,  # Smaller batch for faster response
        use_mmap=True,
        use_mlock=True,  # Lock in memory
        verbose=False,
        n_gpu_layers=32 if os.environ.get("CUDA_VISIBLE_DEVICES") else 0,  # GPU if available
    )
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    llm = None

def quick_pdf_extract(file_content: bytes) -> str:
    """Fast PDF text extraction."""
    try:
        # Try PyMuPDF first (faster)
        pdf_doc = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        # Only read first few pages for speed
        max_pages = min(5, pdf_doc.page_count)
        for i in range(max_pages):
            text += pdf_doc[i].get_text()
        pdf_doc.close()
        return text
    except:
        # Quick PyPDF2 fallback
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        max_pages = min(3, len(pdf_reader.pages))
        for i in range(max_pages):
            text += pdf_reader.pages[i].extract_text()
        return text

def smart_truncate(text: str, question: str, max_chars: int = 1500) -> str:
    """Keep text relevant to question and truncate smartly."""
    
    # Find question keywords
    question_words = set(question.lower().split())
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Score sentences by relevance
    scored_sentences = []
    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue
        sentence_words = set(sentence.lower().split())
        score = len(question_words.intersection(sentence_words))
        scored_sentences.append((score, sentence.strip()))
    
    # Sort by relevance and build context
    scored_sentences.sort(reverse=True)
    
    context = ""
    for score, sentence in scored_sentences:
        if len(context + sentence) < max_chars:
            context += sentence + ". "
        else:
            break
    
    return context.strip() or text[:max_chars]

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    """Upload document and ask question - FAST version."""
    
    if not llm:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read file
        content = await file.read()
        
        # Quick size check
        if len(content) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="File too large")
        
        # Fast text extraction
        if file.filename.lower().endswith('.pdf'):
            text = quick_pdf_extract(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        if len(text.strip()) < 20:
            raise HTTPException(status_code=400, detail="No readable text found")
        
        # Smart truncation based on question relevance
        relevant_text = smart_truncate(text, question)
        
        # Minimal prompt for speed
        prompt = f"Document: {relevant_text}\n\nQ: {question}\nA:"
        
        # Generate answer in thread pool (non-blocking)
        def generate():
            return llm(
                prompt,
                max_tokens=100,  # Shorter answers for speed
                temperature=0.1,  # Lower temperature for faster generation
                top_p=0.8,
                top_k=20,  # Smaller top_k for speed
                repeat_penalty=1.1,
                stop=["\n\n", "Q:", "Document:"],
                echo=False
            )
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, generate)
        
        answer = result["choices"][0]["text"].strip()
        
        return {
            "answer": answer,
            "filename": file.filename,
            "processing_note": "Fast mode - first few pages only"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def health():
    return {"status": "ready", "model": "loaded" if llm else "failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081, workers=1)