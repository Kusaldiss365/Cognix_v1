import re

def extract_accuracy_score(text):
    """
    Extracts an integer accuracy score from text.
    Accepts formats like:
    - 'Accuracy: 85%'
    - 'Accuracy: 85'
    - 'Accuracy: 85.5%'
    """
    match = re.search(r"Accuracy:\s*(\d{1,3}(?:\.\d+)?)(?:\s*%)?", text, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        return min(max(round(score), 0), 100)
    return 0
