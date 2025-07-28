from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil, os

from agents.question_agent import QuestionAgent
from agents.context_agent import ContextAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reflection_agent import ReflectionAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.orchestrator_session import OrchestratorSession
from fastapi.responses import PlainTextResponse

from utils.session_manager import get_or_create_user_and_session

# ==== Config ====
UPLOAD_DIR = "data"
CHROMA_DIR = "./chroma_db"
EVAL_PROMPT = "prompts/evaluation_prompt.txt"
REFLECT_PROMPT = "prompts/reflection_prompt.txt"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== In-Memory Session Store ====
chat_session = {}

class ChatRequest(BaseModel):
    user_id: str | None = None
    user_answer: str
    question_index: int


# ==== Upload Route ====
@app.post("/upload/{session_id}")
async def upload_and_initialize(
    session_id: str,
    notes: UploadFile = File(...),
    questions: UploadFile = File(...),
    answers: UploadFile = File(...)
):
    user_dir = f"data/{session_id}"
    chroma_dir = f"chroma_db/{session_id}"
    os.makedirs(user_dir, exist_ok=True)

    try:
        with open(f"{user_dir}/notes.pdf", "wb") as f:
            shutil.copyfileobj(notes.file, f)
        with open(f"{user_dir}/questions.pdf", "wb") as f:
            shutil.copyfileobj(questions.file, f)
        with open(f"{user_dir}/answers.pdf", "wb") as f:
            shutil.copyfileobj(answers.file, f)

        return JSONResponse(content={"status": "success", "message": "Files uploaded successfully."})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# ==== Chat Route ====
@app.post("/chat/{session_id}")
async def chat_with_orchestrator(session_id: str, req: ChatRequest):
    user_id = req.user_id
    answer = req.user_answer
    index = req.question_index

    # 🔐 Generate a unique key per user per session
    session_id, user_id, unique_session_key = get_or_create_user_and_session(session_id, user_id)

    # 🧠 Create session if not exists
    if unique_session_key not in chat_session:
        base_path = f"data/{session_id}"
        chroma_path = f"chroma_db/{session_id}"

        question_agent = QuestionAgent(f"{base_path}/questions.pdf")
        context_agent = ContextAgent(f"{base_path}/notes.pdf", chroma_path)
        context_agent.ingest_and_index()
        notes_context = context_agent.get_vectorstore().similarity_search("summary")[0].page_content

        evaluation_agent = EvaluationAgent(EVAL_PROMPT, f"{base_path}/answers.pdf", notes_context)
        reflection_agent = ReflectionAgent(REFLECT_PROMPT)

        orchestrator = OrchestratorAgent(question_agent, context_agent, evaluation_agent, reflection_agent)
        chat_session[unique_session_key] = OrchestratorSession(orchestrator, session_id=session_id, user_id=user_id)

    session = chat_session[unique_session_key]

    # 📍 Handle start of session
    if index == 0 and answer.strip() == "":
        return {
            "question": session.get_current_question(),
            "index": 0,
            "total_questions": session.get_total_questions()
        }

    # 📝 Evaluate answer
    result = session.process_answer(answer)

    result["index"] = session.current_index
    result["total_questions"] = session.get_total_questions()

    if session.is_complete():
        result["complete"] = True

    return result



@app.get("/clear-all", response_class=PlainTextResponse)
async def clear_all_sessions_and_data():
    chat_session.clear()
    shutil.rmtree("data", ignore_errors=True)
    shutil.rmtree("chroma_db", ignore_errors=True)
    return "All uploaded data and Chroma vector stores have been cleared."