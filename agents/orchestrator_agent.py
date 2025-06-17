from utils.accuracy_parser import extract_accuracy_score

class OrchestratorAgent:
    def __init__(self, question_agent, context_agent, evaluation_agent, reflection_agent):
        self.q_agent = question_agent
        self.c_agent = context_agent
        self.e_agent = evaluation_agent
        self.r_agent = reflection_agent
        self.vectorstore = context_agent.get_vectorstore()

    def run(self):
        total = self.q_agent.total_questions()

        for i in range(total):
            question = self.q_agent.get_question(i)
            print(f"\n📘 Question {i + 1}/{total}: {question}\n")

            while True:
                user_answer = input("📝 Your Answer (or type 'next' to skip, 'end' to quit): \n").strip()
                print("Loading...")

                if user_answer.lower() == "next":
                    print("➡️ Moving to the next question...")
                    break
                if user_answer.lower() == "end":
                    print("⏹️ Ending session. Goodbye!")
                    return

                # Retrieve context
                context_docs = self.c_agent.retrieve_context(question)
                context_text = "\n\n".join(doc.page_content for doc in context_docs)

                # Evaluate user answer
                question_key = question.split('.')[0].strip()
                reference_answer = self.e_agent.reference_answers.get(question_key, "")
                feedback, accuracy = self.e_agent.evaluate(
                    question=question,
                    user_answer=user_answer,
                    similar_docs=context_docs
                )

                # Reflect on the evaluation feedback
                eval_reflection = self.r_agent.reflect_evaluation(
                    question=question,
                    user_answer=user_answer,
                    expected_answers=reference_answer,
                    notes_context=self.e_agent.notes_context,
                    similar_context=context_text,
                    feedback=feedback
                )

                # Show only what user should see
                # print(f"\n🧾 Feedback:\n{feedback}")
                print(f"\n✅ Accuracy: {accuracy}%")
                print(f"\n💡 Reflection & Hint to Improve:\n{eval_reflection}\n")

                if accuracy >= 75:
                    print("✅ Well done! Moving to the next question...\n")
                    break





