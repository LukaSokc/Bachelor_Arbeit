# eval_open.py

import os
# Transformer-Logging auf ERROR setzen, noch bevor bert_score oder transformers geladen werden
os.environ["TRANSFORMERS_VERBOSITY"]     = "error"
os.environ["TRANSFORMERS_VERBOSITY_API"] = "error"

# Optional noch direkt via API absichern:
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import evaluate
import contextlib, io, nltk

# einmaliges, leises Laden der NLTK-Resourcen
_f = io.StringIO()
with contextlib.redirect_stdout(_f):
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt',   quiet=True)

# einmaliges, leises Laden des METEOR-Metrikmoduls
_f = io.StringIO()
with contextlib.redirect_stdout(_f):
    METEOR = evaluate.load('meteor')

from utils import normalize_answer
from collections import Counter
from bert_score import score as bert_score
import matplotlib.pyplot as plt


def bleu_scores(data, weights):
    smoothie = SmoothingFunction().method4
    return data.apply(
        lambda r: sentence_bleu([r['correct_answer'].split()], r['model_output'].split(),
                                weights=weights, smoothing_function=smoothie), axis=1)


def compute_rouge_scores(data):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = data.apply(lambda r: scorer.score(r['correct_answer'], r['model_output']), axis=1)
    df = pd.DataFrame([
        {
            'rouge1': s['rouge1'].fmeasure,
            'rouge2': s['rouge2'].fmeasure,
            'rougeL': s['rougeL'].fmeasure
        }
        for s in scores
    ])
    return df



def compute_token_f1(data):
    def token_f1(pred, ref):
        # cast to str, falls float oder sonst was
        pred = str(pred)
        ref  = str(ref)
        pred_tokens = pred.split()
        ref_tokens  = ref.split()
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall    = num_same / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)
    return data.apply(lambda r: token_f1(r['model_output'], r['correct_answer']), axis=1)


def compute_bert_scores(data):
    # Ganz sicher: alles zu str casten
    cands = data['model_output'].astype(str).tolist()
    refs  = data['correct_answer'].astype(str).tolist()

    # BERTScore mit bert-base-uncased
    P, R, F1 = bert_score(
        cands=cands,
        refs=refs,
        model_type='bert-base-uncased',  # hier das Modell ändern
        lang='en',
        verbose=False
    )

    return pd.Series({
         'bert_precision': P.mean().item(),
         'bert_recall'   : R.mean().item(),
         'bert_f1'       : F1.mean().item()
     })

def evaluate_combined(path: str):
    data = pd.read_csv(path)
    data = data.dropna(subset=['correct_answer', 'model_output'])
    data['correct_answer'] = data['correct_answer'].astype(str)
    data['model_output']   = data['model_output'].astype(str)
    # Normalize answers
    for col in ['correct_answer', 'model_output']:
        data[col] = data[col].apply(normalize_answer)

    # Summary
    print("Combined Evaluation")
    print("=" * 30)
    print(f"Samples evaluated: {len(data)}")

    # BLEU scores
    data['bleu1'] = bleu_scores(data, (1, 0, 0, 0))
    print(f"BLEU-1: {data['bleu1'].mean():.4f}")
    data['bleu2'] = bleu_scores(data, (0.5, 0.5, 0, 0))
    print(f"BLEU-2: {data['bleu2'].mean():.4f}")
    data['bleu4'] = bleu_scores(data, (0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-4: {data['bleu4'].mean():.4f}")
    print("=" * 30)

    # ROUGE scores
    rouge_df = compute_rouge_scores(data)
    data = pd.concat([data, rouge_df], axis=1)
    print(f"ROUGE-1: {data['rouge1'].mean():.4f}")
    print(f"ROUGE-2: {data['rouge2'].mean():.4f}")
    print(f"ROUGE-L: {data['rougeL'].mean():.4f}")
    print("=" * 30)

    # Token-level F1
    data['token_f1'] = compute_token_f1(data)
    print(f"Token-F1: {data['token_f1'].mean():.4f}")
    print("=" * 30)

    # BERTScore
    bert_pd = compute_bert_scores(data)
    print(f"BERTScore-precision: {bert_pd['bert_precision']:.4f}")
    print(f"BERTScore-recall: {bert_pd['bert_recall']:.4f}")
    print(f"BERTScore-F1: {bert_pd['bert_f1']:.4f}")
    print("=" * 30)