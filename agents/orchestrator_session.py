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

    def get_total_questions(self):
        return self.orchestrator.q_agent.total_questions()

    def is_complete(self):
        return self.current_index >= self.orchestrator.q_agent.total_questions()

    def get_final_summary(self):
        avg_accuracy = sum(r["accuracy"] for r in self.results) / len(self.results) if self.results else 0
        low_scores = [r for r in self.results if r["accuracy"] < 80]

        final_summary = self.orchestrator.r_agent.generate_final_summary(
            avg_accuracy=avg_accuracy,
            weak_points=[r["question"] for r in low_scores],
            notes_context=self.orchestrator.e_agent.notes_context
        )

        return final_summary

    def process_answer(self, user_answer: str):
        q = self.orchestrator.q_agent
        c = self.orchestrator.c_agent
        e = self.orchestrator.e_agent
        r = self.orchestrator.r_agent

        # --- 0. Start session ---
        if user_answer == "[START_SESSION]":
            return {
                "message": "Welcome! Here's your first question:",
                "index": self.current_index,
                "question": self.get_current_question(),
                "complete": False,
                "retry": False
            }

        # --- 1. End chat ---
        if user_answer == "[END_CHAT]":
            return {
                "message": "Chat ended. Thank you.",
                "index": self.current_index,
                "question": None,
                "complete": True,
                "retry": False
            }

        # --- 2. Next question ---
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

        # --- 2.5. Hint request ---
        if user_answer == "[GET_HINT_ONLY]":
            try:
                question_text = q.get_question(self.current_index)
                context_docs = c.retrieve_context(question_text)

                if not context_docs:
                    return {
                        "message": "⚠️ No relevant material found to generate a hint.",
                        "index": self.current_index,
                        "question": None,
                        "complete": False,
                        "retry": True
                    }

                similar_context = "\n\n".join(getattr(doc, "page_content", "") for doc in context_docs)
                hint = e.generate_direct_hint(question_text, similar_context)

                return {
                    "message": hint,
                    "index": self.current_index,
                    "question": None,
                    "complete": False,
                    "retry": True
                }

            except Exception as err:
                print(f"❌ [GET_HINT_ONLY] failed: {err}")
                return {
                    "message": "⚠️ Failed to generate hint due to internal error.",
                    "index": self.current_index,
                    "question": None,
                    "complete": False,
                    "retry": True
                }

        # --- 3. Regular answer ---
        question_text = q.get_question(self.current_index)
        question_number = q.get_question_number(self.current_index)
        context_docs = c.retrieve_context(question_text)

        raw_feedback, accuracy, reference_answer = e.evaluate(
            question_number, question_text, user_answer, context_docs
        )
        accuracy = min(max(int(accuracy), 0), 100) if accuracy is not None else 0
        feedback = raw_feedback.split("Feedback:")[-1].strip() if "Feedback:" in raw_feedback else raw_feedback

        if accuracy == 100:
            reflection = "Great job — your answer fully addresses the question and shows solid understanding. Keep it up!"
        else:
            reflection = r.reflect_evaluation(
                question=question_text,
                user_answer=user_answer,
                expected_answers=reference_answer,
                notes_context=e.notes_context,
                similar_context="\n\n".join(getattr(doc, "page_content", "") for doc in context_docs),
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
            response["final_summary"] = self.get_final_summary()

        return response
