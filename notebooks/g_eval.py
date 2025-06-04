import os, json, re, string                     # <-- string import
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

data = pd.read_csv('data/llm_answers/qwen_batchsize_8.csv')
data.index = data.index + 1          # <<< NEU
# Openended Questions
def normalize_answer(ans):
    ans = ans.lower()  # Kleinbuchstaben
    ans = re.sub(r'^answer:\s*', '', ans)  # Entferne "answer:" am Anfang, evtl. mit Leerzeichen danach
    ans = re.sub(f"[{re.escape(string.punctuation)}]", "", ans)  # Entferne Satzzeichen
    ans = ans.strip()  # Entferne führende und folgende Leerzeichen
    return ans
data_open = data[~data['correct_answer'].isin(['yes', 'no'])].copy()
data_open['correct_answer'] = data_open['correct_answer'].apply(normalize_answer)
data_open['model_output']   = data_open['model_output'].apply(normalize_answer)
print(len(data_open))

# ---------- Konfiguration ----------
output_csv = "data/G_eval/Qwen_2.5/batch_size_8_g_eval_run_2.csv"
csv_cols   = ["index", "g_eval_score", "g_eval_judgment",
              "g_eval_expl", "g_eval_raw"]

client = OpenAI(api_key="")  # or use env var


def _round_to_tenth(x: float) -> float:
    return round(round(x * 10) / 10, 1)

# ---------- Evaluationsfunktion ----------
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

        # Kostenschätzung
        if model_name == "gpt-4o-mini":
            cost = (usage.prompt_tokens * 0.01 +
                    usage.completion_tokens * 0.03) / 1000
        elif model_name == "gpt-3.5-turbo":
            cost = (usage.prompt_tokens * 0.001 +
                    usage.completion_tokens * 0.002) / 1000
        else:
            cost = 0

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

# ---------- CSV vorbereiten ----------
if not os.path.exists(output_csv):
    pd.DataFrame(columns=csv_cols).to_csv(output_csv, index=False)

evaluated_idx = set(pd.read_csv(output_csv)["index"].tolist())

# ---------- Hauptschleife ----------
total_cost = 0.0

for row in tqdm(data_open.itertuples(), total=len(data_open)):
    idx = row.Index                              # <-- richtiger Index
    if idx in evaluated_idx:
        continue
    res = g_eval_dual(row.question,
                      row.model_output,
                      row.correct_answer)
    total_cost += res["tokens"]["cost"]
    # sofort anhängen & flushen
    with open(output_csv, "a", newline='', encoding='utf-8') as f:
        pd.DataFrame([{
            "index": idx,
            "g_eval_score": res["score"],
            "g_eval_judgment": res["judgment"],
            "g_eval_expl": res["explanation"],
            "g_eval_raw": res["raw_output"]
        }]).to_csv(f, header=False, index=False)
        f.flush()

print(f"\n🔥 Total evaluation cost this run: ${total_cost:.4f}\n")