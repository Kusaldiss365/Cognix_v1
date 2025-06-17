from langchain_ollama import OllamaLLM

from utils.accuracy_parser import extract_accuracy_score
from utils.pdf_loader import load_pdf_text


class EvaluationAgent:
    def __init__(self, prompt_path, answers_pdf_path, notes_context):
        with open(prompt_path, "r") as file:
            self.prompt_template = file.read()
        full_text = load_pdf_text(answers_pdf_path)
        self.reference_answers = self.parse_answers(full_text)
        self.notes_context = notes_context
        self.llm = OllamaLLM(model="llama3")

    def parse_answers(self, text):
        # Simple parse based on questions numbered like "1. Answer text"
        answers = {}
        current_key = None
        current_answer_lines = []

        for line in text.splitlines():
            line = line.strip()
            if line == "":
                continue
            # Detect if line starts with a number and dot, e.g. "1."
            if line[0].isdigit() and line[1] == '.':
                if current_key is not None:
                    # Save previous answer
                    answers[current_key] = " ".join(current_answer_lines).strip()
                    current_answer_lines = []
                current_key = line.split('.')[0]  # e.g. "1"
                current_answer_lines.append(line[len(current_key)+1:].strip())
            else:
                # Continuation of answer text
                current_answer_lines.append(line)

        # Save last answer
        if current_key is not None:
            answers[current_key] = " ".join(current_answer_lines).strip()

        return answers

    def evaluate(self, question, user_answer, similar_docs):
        # Extract question number
        question_key = question.split('.')[0].strip()
        reference_answer = self.reference_answers.get(question_key, "")

        context_text = "\n\n".join([doc.page_content for doc in similar_docs])
        prompt = self.prompt_template.format(
            question=question,
            user_answer=user_answer,
            expected_answers=reference_answer,
            notes_context=self.notes_context,
            similar_context=context_text
        )

        response = self.llm.invoke(prompt)

        # Extract feedback and accuracy
        feedback = None
        for line in response.splitlines():
            if line.lower().startswith("feedback:"):
                feedback = line[len("feedback:"):].strip()

        accuracy = extract_accuracy_score(response)

        if feedback is None:
            feedback = response  # fallback if not found

        return feedback, accuracy


