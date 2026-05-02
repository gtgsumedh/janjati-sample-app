from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse 
import os
import uuid
from datetime import datetime
import logging

# Set up main logging
logging.basicConfig(
    filename='janjati_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our custom modules
from database import init_db, save_session
from schemas import InitRequest, QuestionRequest
from vector_db import find_best_context
from llm_service import get_llm_answer

# Initialize Database
init_db()

app = FastAPI(title="Janjati Knowledge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve the HTML Frontend ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# --- API 1: INIT ---
@app.post("/api/init")
def init_lms(request: InitRequest):
    uid = str(uuid.uuid4())
    logger.info(f"Init Request - Username: {request.username}, Method: {request.method}")

    # Save to SQLite using database.py
    save_session(uid, request.username, request.reqdatetime)

    return {
        "method": "init_lms_res",
        "username": request.username,
        "uniqueid": uid,
        "reqdatetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# --- API 2: QUESTION ---
@app.post("/api/ask")
def ask_question(request: QuestionRequest):
    logger.info(f"Question Request - Username: {request.username}, UID: {request.uniqueid}, Question: {request.question}")

    # Step 1: Use Vector DB to find context
    best_context = find_best_context(request.question)
    
    # Step 2: Use LLM Service to generate answer
    bot_reply = get_llm_answer(best_context, request.question)

    return {
        "method": "init_lms_answer",
        "username": request.username,
        "response": bot_reply,
        "reqdatetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }