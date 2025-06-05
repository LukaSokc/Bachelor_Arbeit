import os, json, re, string                     # <-- string import
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ---------------------------- Daten laden ---------------------------- #
data = pd.read_csv('docs/gemma/fine_tuned_batchsize_2_test_set.csv')
data.index = data.index + 1          
# ------------------ Hilfsfunktion zur Normalisierung ----------------- #
def normalize_answer(ans: str) -> str:
    """In Kleinbuchstaben umwandeln, 'answer:' entfernen, Satzzeichen entfernen, trimmen."""
    ans = ans.lower()
    ans = re.sub(r"^answer:\s*", "", ans)
    ans = re.sub(f"[{re.escape(string.punctuation)}]", "", ans)
    return ans.strip()

data_open = data[~data['correct_answer'].isin(['yes', 'no'])].copy()
data_open['correct_answer'] = data_open['correct_answer'].apply(normalize_answer)
data_open['model_output']   = data_open['model_output'].apply(normalize_answer)

# ----------------------------- Konfiguration ------------------------------ #
output_csv = "docs/gemma/g_eval_runs/batch_size_2_test_run_g_eval_run_1.csv"
csv_cols   = ["index", "g_eval_score", "g_eval_judgment",
              "g_eval_expl", "g_eval_raw"]

client = OpenAI(api_key="") 


def round_to_tenth(x: float) -> float:
    """Rundet auf eine Stelle nach dem Komma."""
    return round(round(x * 10) / 10, 1)

# --------------------------- Evaluationsfunktion --------------------------- #
def g_eval_dual(question, model_output, correct_answer,
                model_name="gpt-4o-mini"):
    prompt = f"""You are a medical expert evaluating an AI assistant that answers open-ended medical questions based on images.

    Assess whether the model's answer has the same medical meaning as the correct answer. Focus on clinical acceptability and semantic equivalence.

    Return **only** valid JSON with these keys:
    {{
    "explanation": "<one concise sentence, max 25 words>",
    "score": <float 0.0-1.0, step 0.1>,
    "judgment": "<Correct|Incorrect>"
    }}

    Scoring Guide
    - 1.0          Medically accurate and equivalent
    - 0.9-0.6      Minor discrepancies, still acceptable
    - 0.5-0.3      Partially correct; could mislead
    - 0.2-0.1      Vague or largely incomplete
    - 0.0          Wrong or not relevant

    Question: {question}

    Model Answer: {model_output}
    Correct Answer: {correct_answer}
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        output_text = response.choices[0].message.content
        usage       = response.usage

        # JSON-Parsing
        try:
            data = json.loads(output_text)
            score    = _round_to_tenth(float(data["score"]))
            judgment = data["judgment"].capitalize()
            expl     = data["explanation"]
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback per Regex
            score_match    = re.search(r"Score:\s*([0-9.]+)", output_text)
            judgment_match = re.search(r"Judgment:\s*(Correct|Incorrect)",
                                       output_text, re.IGNORECASE)
            score    = _round_to_tenth(float(score_match.group(1))) if score_match else None
            judgment = judgment_match.group(1).capitalize() if judgment_match else "Unknown"
            expl     = None

        return {
            "score": score, "judgment": judgment, "explanation": expl,
            "raw_output": output_text,
            "tokens": {"prompt": usage.prompt_tokens,
                       "completion": usage.completion_tokens,
                       "total": usage.total_tokens,
                       "cost": cost}
        }
    except Exception as e:
        return {
            "score": None, "judgment": "Error", "explanation": None,
            "raw_output": str(e),
            "tokens": {"prompt": 0, "completion": 0, "total": 0, "cost": 0}
        }

# ------------------------- Ausgabe-CSV vorbereiten ------------------------- #
if not os.path.exists(output_csv):
    pd.DataFrame(columns=csv_cols).to_csv(output_csv, index=False)

evaluated_idx = set(pd.read_csv(output_csv)["index"].tolist())

# ----------------------------- Hauptschleife ------------------------------ #
for row in tqdm(data_open.itertuples(), total=len(data_open)):
    idx = row.Index                          
    if idx in evaluated_idx:
        continue
    res = g_eval_dual(row.question,
                      row.model_output,
                      row.correct_answer)
    total_cost += res["tokens"]["cost"]
    # Ergebnis sofort ans Ende der CSV anhaengen
    with open(output_csv, "a", newline='', encoding='utf-8') as f:
        pd.DataFrame([{
            "index": idx,
            "g_eval_score": res["score"],
            "g_eval_judgment": res["judgment"],
            "g_eval_expl": res["explanation"],
            "g_eval_raw": res["raw_output"]
        }]).to_csv(f, header=False, index=False)
        f.flush()

print(f"\n Total evaluation cost this run: ${total_cost:.4f}\n")