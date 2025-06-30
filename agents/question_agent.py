import re
from utils.pdf_loader import load_pdf_text
from langchain_ollama import OllamaLLM

class QuestionAgent:
    def __init__(self, question_pdf_path):
        raw_text = load_pdf_text(question_pdf_path)
        self.llm = OllamaLLM(model="llama3")

        prompt = (
            "You are a helpful assistant. Extract all clear and complete questions "
            "from the text below. Ignore headings or explanations. "
            "Return ONLY the questions as a numbered list (1., 2., 3., etc). "
            "Each question must start with its number and a dot.\n\n"
            f"{raw_text}\n\n"
            "Questions:"
        )

        extracted = self.llm.invoke(prompt)

        # Strict regex: match only lines starting with number + dot + space
        pattern = re.compile(r"^\s*(\d+)\.\s+(.+)")

        self.questions = []
        for line in extracted.splitlines():
            match = pattern.match(line)
            if match:
                number = int(match.group(1))
                text = match.group(2).strip()
                self.questions.append( (number, text) )
            else:
                print(f"Skipping non-matching line: {line}")

    def get_question(self, index):
        return self.questions[index][1] if index < len(self.questions) else None

    def get_question_number(self, index):
        return self.questions[index][0] if index < len(self.questions) else None

    def total_questions(self):
        return len(self.questions)
