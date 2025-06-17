from utils.pdf_loader import load_pdf_text

class QuestionAgent:
    def __init__(self, question_pdf_path):
        self.questions = load_pdf_text(question_pdf_path).split("\n")

    def get_question(self, index):
        return self.questions[index].strip() if index < len(self.questions) else None

    def total_questions(self):
        return len(self.questions)