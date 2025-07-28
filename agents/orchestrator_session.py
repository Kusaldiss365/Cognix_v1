from agents.orchestrator_agent import OrchestratorAgent
import uuid

class OrchestratorSession:
    def __init__(self, orchestrator: OrchestratorAgent, session_id=None, user_id=None):
        self.orchestrator = orchestrator
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id or str(uuid.uuid4())
        self.current_index = 0
        self.results = []
        self.waiting_for_retry = False

    def get_current_question(self):
        if self.current_index >= self.orchestrator.q_agent.total_questions():
            return None
        return self.orchestrator.q_agent.get_question(self.current_index)

    def process_answer(self, user_answer: str):
        q = self.orchestrator.q_agent
        c = self.orchestrator.c_agent
        e = self.orchestrator.e_agent
        r = self.orchestrator.r_agent

        # --- 0. Start session from frontend ---
        if user_answer == "[START_SESSION]":
            return {
                "message": "Welcome! Here's your first question:",
                "index": self.current_index,
                "question": self.get_current_question(),
                "complete": False,
                "retry": False
            }

        # --- 1. End chat request from frontend ---
        if user_answer == "[END_CHAT]":
            return {
                "message": "Chat ended. Thank you.",
                "index": self.current_index,
                "question": None,
                "complete": True,
                "retry": False
            }

        # --- 2. Next question request from frontend ---
        if user_answer == "[NEXT_QUESTION]":
            self.current_index += 1
            next_question = self.get_current_question()
            done = self.current_index >= q.total_questions()

            return {
                "message": f"Question {self.current_index + 1} of {q.total_questions()}:<br>{next_question}" if next_question else "All questions completed.",
                "index": self.current_index,
                "question": next_question,
                "complete": done,
                "retry": False
            }

        # --- 3. Normal answer flow ---
        question_text = q.get_question(self.current_index)
        question_number = q.get_question_number(self.current_index)

        context_docs = c.retrieve_context(question_text)
        reference_answer = e.reference_answers.get(question_number, "")

        # Dynamically generate reference answer if missing
        if not reference_answer:
            context_text = "\n\n".join(doc.page_content for doc in context_docs)
            gen_prompt = (
                f"Generate a complete and factual answer for the following question using only the given context:\n\n"
                f"Question: {question_text}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Answer:"
            )
            reference_answer = e.llm.invoke(gen_prompt).strip()
            e.reference_answers[question_number] = reference_answer

        raw_feedback, accuracy = e.evaluate(question_number, question_text, user_answer, context_docs)
        accuracy = min(max(int(accuracy), 0), 100) if accuracy is not None else 0
        feedback = raw_feedback.split("Feedback:")[-1].strip() if "Feedback:" in raw_feedback else raw_feedback

        # Only reflect if accuracy < 100
        if accuracy == 100:
            reflection = "Great job! Your answer was complete. No improvements needed."
        else:
            reflection = r.reflect_evaluation(
                question=question_text,
                user_answer=user_answer,
                expected_answers=reference_answer,
                notes_context=e.notes_context,
                similar_context="\n\n".join(doc.page_content for doc in context_docs),
                feedback=feedback
            )

        self.results.append({
            "question": question_text,
            "user_answer": user_answer,
            "accuracy": accuracy,
            "feedback": feedback
        })

        if accuracy > 80:
            self.current_index += 1
            self.waiting_for_retry = False
        else:
            self.waiting_for_retry = True

        next_question = self.get_current_question()
        done = self.current_index >= q.total_questions()

        response = {
            "message": f"Accuracy: {accuracy}%<br/>Hint:<br/>{reflection}",
            "index": self.current_index,
            "question": next_question,
            "complete": done,
            "retry": self.waiting_for_retry
        }

        if done:
            summary_html = self.orchestrator.provide_overall_feedback(self.results)
            response["final_summary"] = summary_html

        return response

    def is_complete(self):
        return self.current_index >= self.orchestrator.q_agent.total_questions()

    def get_summary(self):
        return self.orchestrator.provide_overall_feedback(self.results)

    def get_total_questions(self):
        return self.orchestrator.q_agent.total_questions()

