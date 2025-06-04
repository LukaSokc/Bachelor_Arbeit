# utils.py
import re, string

def normalize_answer(ans: str) -> str:
    """Kleinbuchstaben, remove punctuation und 'answer:' prefix."""
    ans = ans.lower().strip()
    ans = re.sub(r'^answer:\s*', '', ans)
    ans = re.sub(f"[{re.escape(string.punctuation)}]", "", ans)
    return ans

def load_csv(path: str):
    import pandas as pd
    return pd.read_csv(path)
