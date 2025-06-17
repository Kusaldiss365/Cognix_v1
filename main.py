from agents.question_agent import QuestionAgent
from agents.context_agent import ContextAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reflection_agent import ReflectionAgent
from agents.orchestrator_agent import OrchestratorAgent

material_path = "data/notes.pdf"
question_path = "data/questions.pdf"
answer_path = "data/answers.pdf"
eval_prompt_path = "prompts/evaluation_prompt.txt"
reflect_prompt_path = "prompts/reflection_prompt.txt"
chroma_dir = "./chroma_db"

# Setup agents
question_agent = QuestionAgent(question_path)
context_agent = ContextAgent(material_path, chroma_dir)
context_agent.ingest_and_index()
notes_context = context_agent.get_vectorstore().similarity_search("summary")[0].page_content

evaluation_agent = EvaluationAgent(eval_prompt_path, answer_path, notes_context)
reflection_agent = ReflectionAgent(reflect_prompt_path)
orchestrator = OrchestratorAgent(question_agent, context_agent, evaluation_agent, reflection_agent)

# Start chat loop
orchestrator.run()
