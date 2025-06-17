import re

def extract_accuracy_score(text):
    match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
    if match:
        score = float(match.group(1))
        return min(max(round(score), 0), 100)
    return 0


