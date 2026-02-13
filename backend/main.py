# main.py
from agents.question_agent import QuestionAgent
from agents.context_agent import ContextAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reflection_agent import ReflectionAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.orchestrator_session import OrchestratorSession
from uuid import uuid4

def create_orchestrator_session(session_id: str, user_id: str | None = None):
    base_path = f"data/{session_id}"
    chroma_dir = f"chroma_db/{session_id}"

    question_path = f"{base_path}/questions.pdf"
    material_path = f"{base_path}/notes.pdf"
    answer_path = f"{base_path}/answers.pdf"
    eval_prompt_path = "prompts/evaluation_prompt.txt"
    reflect_prompt_path = "prompts/reflection_prompt.txt"

    # Setup agents
    question_agent = QuestionAgent(question_path)
    context_agent = ContextAgent(material_path, chroma_dir)
    context_agent.ingest_and_index()
    notes_context = context_agent.get_vectorstore().similarity_search("summary")[0].page_content

    evaluation_agent = EvaluationAgent(eval_prompt_path, answer_path, notes_context)
    reflection_agent = ReflectionAgent(reflect_prompt_path)
    orchestrator = OrchestratorAgent(question_agent, context_agent, evaluation_agent, reflection_agent)

    return OrchestratorSession(orchestrator, session_id=session_id, user_id=user_id or str(uuid4()))

