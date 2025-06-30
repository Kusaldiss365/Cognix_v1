import os
import re
from langchain_ollama import OllamaLLM
from utils.accuracy_parser import extract_accuracy_score
from utils.pdf_loader import load_pdf_text

class EvaluationAgent:
    def __init__(self, prompt_path, answers_pdf_path, notes_context):
        # Load evaluation prompt template
        with open(prompt_path, "r") as file:
            self.prompt_template = file.read()

        self.llm = OllamaLLM(model="llama3")

        self.reference_answers = {}
        if os.path.exists(answers_pdf_path):
            print(f"Found {answers_pdf_path}. Parsing reference answers...")
            full_text = load_pdf_text(answers_pdf_path)
            self.reference_answers = self.parse_answers(full_text)
        else:
            print(f"{answers_pdf_path} not found! Reference answers will be generated on-the-fly.")

        self.notes_context = notes_context

    def parse_answers(self, text):
        """
        Use Llama3 to robustly extract numbered answers.
        Merges multi-line answers into single lines.
        """
        prompt = (
            "You are a helpful assistant. Extract all numbered answers from the following text. "
            "Each answer must be complete and self-contained. If an answer spans multiple lines, merge it into a single line. "
            "Example:\n\n"
            "Input:\n1. Android Inc. was founded in 2003 by Andy Rubin, Rich Miner, Nick Sears, and Chris White.\n"
            "The company initially aimed to develop an advanced OS for cameras.\n\n"
            "Output:\n1. Android Inc. was founded in 2003 by Andy Rubin, Rich Miner, Nick Sears, and Chris White. The company initially aimed to develop an advanced OS for cameras.\n\n"
            f"Now process the following text:\n{text}\n\nAnswers:"
        )

        extracted = self.llm.invoke(prompt)

        # Parse: match lines starting with number dot space
        pattern = re.compile(r"^\s*(\d+)\.\s+(.+)")
        answers = {}
        current_number = None
        current_lines = []

        for line in extracted.splitlines():
            match = pattern.match(line)
            if match:
                if current_number is not None:
                    answers[current_number] = " ".join(current_lines).strip()
                current_number = int(match.group(1))  # INT key
                current_lines = [match.group(2).strip()]
            elif current_number:
                current_lines.append(line.strip())

        if current_number is not None:
            answers[current_number] = " ".join(current_lines).strip()

        print("Parsed Reference Answers:")
        for k, v in answers.items():
            print(f"{k}: {v}")

        return answers

    def evaluate(self, question_number, question_text, user_answer, similar_docs):
        """
        Evaluate user's answer against the reference answer.
        If reference answer is missing, generate it dynamically.
        """
        reference_answer = self.reference_answers.get(question_number, "")

        if not reference_answer:
            print(f"No reference answer found for Q{question_number}. Generating dynamically...")
            context_text = "\n\n".join(doc.page_content for doc in similar_docs)
            gen_prompt = (
                f"Generate a complete and factual answer for the following question using only the given context:\n\n"
                f"Question: {question_text}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Answer:"
            )
            reference_answer = self.llm.invoke(gen_prompt).strip()
            self.reference_answers[question_number] = reference_answer
            # print(f"Generated Reference Answer: {reference_answer}")

        # Shortcut: exact match
        if user_answer.strip().lower() == reference_answer.strip().lower():
            return "Feedback: Perfect match! Your answer is exactly correct.", 100

        context_text = "\n\n".join(doc.page_content for doc in similar_docs)

        # Fill prompt
        prompt = self.prompt_template.format(
            question=question_text,
            user_answer=user_answer,
            expected_answers=reference_answer,
            notes_context=self.notes_context,
            similar_context=context_text
        )

        # print("\n=== EVAL PROMPT ===\n")
        # print(prompt)

        response = self.llm.invoke(prompt)

        # print("\n=== LLM RESPONSE ===\n")
        # print(response)

        # Parse feedback and accuracy
        feedback = None
        for line in response.splitlines():
            if line.lower().startswith("feedback:"):
                feedback = line[len("feedback:"):].strip()

        accuracy = extract_accuracy_score(response)
        try:
            accuracy = int(accuracy)
            accuracy = min(max(accuracy, 0), 100)
        except:
            accuracy = 0

        if feedback is None:
            feedback = response  # fallback

        return feedback, accuracy
