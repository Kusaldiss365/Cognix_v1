from langchain_ollama import OllamaLLM


class ReflectionAgent:
    def __init__(self, prompt_path):
        with open(prompt_path, "r") as file:
            self.prompt_template = file.read()
        self.llm = OllamaLLM(model="llama3")

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
