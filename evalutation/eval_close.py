# eval_close.py
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from utils import normalize_answer

def evaluate_close(path: str):
    data = pd.read_csv(path)
    # Filter only yes/no
    data = data[data['correct_answer'].isin(['yes','no']) & data['model_output'].isin(['yes','no'])]
    # Normalize
    for col in ['correct_answer','model_output']:
        data[col] = data[col].apply(normalize_answer)
    # Metriken
    acc = (data['correct_answer']==data['model_output']).mean()
    f1 = f1_score(data['correct_answer'], data['model_output'], average='macro')
    prec = precision_score(data['correct_answer'], data['model_output'], average='macro')
    rec = recall_score(data['correct_answer'], data['model_output'], average='macro')
    print(f"[Close] Accuracy: {acc:.2%}, Precision: {prec:.2%}, Recall: {rec:.2%}, F1: {f1:.2%}")
    print(classification_report(data['correct_answer'], data['model_output']))
