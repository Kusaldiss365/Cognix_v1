import textwrap

class OrchestratorAgent:
    def __init__(self, question_agent, context_agent, evaluation_agent, reflection_agent):
        self.q_agent = question_agent
        self.c_agent = context_agent
        self.e_agent = evaluation_agent
        self.r_agent = reflection_agent
        self.vectorstore = context_agent.get_vectorstore()

    def run(self):
        total = self.q_agent.total_questions()
        results = []  #store tuples (question, user_answer, accuracy)

        for i in range(total):
            question_text = self.q_agent.get_question(i)
            question_number = self.q_agent.get_question_number(i)

            print(f"\nQuestion {i + 1}/{total}: {question_text}\n")

            while True:
                user_answer = input("Your Answer (or type 'next' to skip, 'end' to quit): ").strip()
                print("\nLoading...")

                if user_answer.lower() == "next":
                    print("Moving to the next question...")
                    break
                if user_answer.lower() == "end":
                    print("Ending session. Goodbye!")
                    return

                context_docs = self.c_agent.retrieve_context(question_text)

                # print("Top k=3: ",context_docs)

                reference_answer = self.e_agent.reference_answers.get(question_number, "")
                raw_feedback, accuracy = self.e_agent.evaluate(
                    question_number,
                    question_text,
                    user_answer,
                    context_docs
                )

                accuracy = min(max(int(accuracy), 0), 100) if accuracy is not None else 0

                feedback = raw_feedback if "Feedback:" not in raw_feedback else raw_feedback.split("Feedback:")[
                    -1].strip()

                eval_reflection = self.r_agent.reflect_evaluation(
                    question=question_text,
                    user_answer=user_answer,
                    expected_answers=reference_answer,
                    notes_context=self.e_agent.notes_context,
                    similar_context="\n\n".join(doc.page_content for doc in context_docs),
                    feedback=feedback
                )

                print(f"\nAccuracy: {accuracy}%")
                print(f"\nReflection & Hint to Improve:\n{textwrap.fill(eval_reflection, width=80)}\n")

                # Save result
                results.append({
                    "question": question_text,
                    "user_answer": user_answer,
                    "accuracy": accuracy,
                    "feedback": feedback
                })

                if accuracy > 80:
                    print("Well done! Moving to the next question...\n")
                    break

        # After all questions:
        self.provide_overall_feedback(results)

    def provide_overall_feedback(self, results):
        print("\n=== Overall Performance Summary ===\n")

        # Calculate average accuracy
        if results:
            avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
        else:
            avg_accuracy = 0

        print(f"Average Accuracy: {avg_accuracy:.1f}%")

        # Identify common gaps
        low_scores = [r for r in results if r["accuracy"] < 80]
        if low_scores:
            print("\nYou may want to review the following topics:\n")
            for r in low_scores:
                print(f"• Q: {r['question']}")
                print(f"  Feedback: {r['feedback']}\n")
        else:
            print("\nExcellent! All answers were well above the passing mark.\n")

        # Use ReflectionAgent for a final motivational wrap-up
        final_summary = self.r_agent.generate_final_summary(
            avg_accuracy=avg_accuracy,
            weak_points=[r["question"] for r in low_scores],
            notes_context=self.e_agent.notes_context
        )

        print("\n=== Final Guidance ===\n")
        print(textwrap.fill(final_summary, width=80))

        return final_summary

