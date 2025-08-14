from agents.orchestrator_agent import OrchestratorAgent
import uuid
import random
import re  # NEW

ENCOURAGING_STARTERS = [
    "Good effort! To improve, check “{title}” (Page {page}) and ",
    "Nice work so far! Review “{title}” (Page {page}) and ",
    "You're on the right track — explore “{title}” (Page {page}) to ",
    "Well done getting this far! Strengthen your understanding by revisiting “{title}” (Page {page}) and ",
    "Keep going strong! Have a look at “{title}” (Page {page}) for more insights and ",
    "Solid progress! Dive into “{title}” (Page {page}) to deepen your understanding and ",
]

class OrchestratorSession:
    def __init__(self, orchestrator: OrchestratorAgent, session_id=None, user_id=None):
        self.orchestrator = orchestrator
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id or str(uuid.uuid4())
        self.current_index = 0
        self.results = []
        self.waiting_for_retry = False

    # --- NEW: small helper to clean noisy titles like "Digital Forensics | Mr. ..."
    def _clean_title(self, title: str) -> str:
        if not title:
            return "Untitled"
        # keep left side before | / – / —
        title = re.split(r"\s*[|–—]\s*", title, maxsplit=1)[0].strip()
        return (title[:60] + "…") if len(title) > 60 else title

    # --- NEW: get first doc's page + title
    def _first_page_and_title(self, docs):
        if not docs:
            return None, None
        page = (docs[0].metadata.get("page") or 0) + 1
        title = self._clean_title(docs[0].metadata.get("page_title", "Untitled"))
        return page, title

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

                similar_context = c.format_hits_with_citations(context_docs)
                hint = e.generate_direct_hint(question_text, similar_context)

                # ---------- NEW: prepend random encouraging starter with page + title
                page_num, title = self._first_page_and_title(context_docs)
                if page_num and hint:
                    starter = random.choice(ENCOURAGING_STARTERS).format(title=title, page=page_num)
                    hint = starter + hint[0].lower() + hint[1:]

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
                similar_context=c.format_hits_with_citations(context_docs),
                feedback=feedback
            )

            # ---------- NEW: prepend random encouraging starter with page + title
            if accuracy > 0:
                page_num, title = self._first_page_and_title(context_docs)
                if page_num and reflection:
                    starter = random.choice(ENCOURAGING_STARTERS).format(title=title, page=page_num)
                    reflection = starter + reflection[0].lower() + reflection[1:]

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

        # -------- CHANGED: remove the old page_hint header; the page is now inside the reflection/hint opener
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
