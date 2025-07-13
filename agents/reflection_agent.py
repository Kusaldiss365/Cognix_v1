from utils.openai_config import get_openai_llm


class ReflectionAgent:
    def __init__(self, prompt_path):
        with open(prompt_path, "r") as file:
            self.prompt_template = file.read()
        self.llm = get_openai_llm()

    def reflect_evaluation(self, question, user_answer, expected_answers, notes_context, similar_context, feedback):
        prompt = self.prompt_template.format(
            question=question,
            user_answer=user_answer,
            expected_answers=expected_answers,
            notes_context=notes_context,
            similar_context=similar_context,
            feedback=feedback
        )
        return self.llm.invoke(prompt)

class ReflectionAgent:
    def __init__(self, prompt_path):
        with open(prompt_path, "r") as file:
            self.prompt_template = file.read()
        self.llm = get_openai_llm()

    def reflect_evaluation(self, question, user_answer, expected_answers, notes_context, similar_context, feedback):
        prompt = self.prompt_template.format(
            question=question,
            user_answer=user_answer,
            expected_answers=expected_answers,
            notes_context=notes_context,
            similar_context=similar_context,
            feedback=feedback
        )
        return self.llm.invoke(prompt).content

    def generate_final_summary(self, avg_accuracy, weak_points, notes_context):
        prompt = (
            "You are an educational assistant. Given the average accuracy and the list of weak topics, "
            "give the student an encouraging final summary, with concrete study advice, in under 100 words. "
            "Mention any specific topics to review if provided.\n\n"
            f"Average Accuracy: {avg_accuracy:.1f}%\n"
            f"Weak Topics: {weak_points}\n\n"
            f"Lecture Notes Context: {notes_context}\n\n"
            "Your summary:"
        )
        return self.llm.invoke(prompt).content
