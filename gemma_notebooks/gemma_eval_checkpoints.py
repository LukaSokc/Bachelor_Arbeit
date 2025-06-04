# ===== 0. Bibliotheken =====
import csv
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

# ===== 1. Authentifizierung (Hugging Face) =====
login("hf_login")

# ===== 2. Verzeichnisse & Dateien =====
PROJECT_ROOT = Path.cwd().parent
MODEL_ROOT = PROJECT_ROOT / "gemma-product-description" / "gemma-product-description"
OUT_CSV = PROJECT_ROOT / "docs" / "gemma" / "gemma_checkpoints.csv"
INDICES_FILE = PROJECT_ROOT / "this_studio" / "validation_subset_indices.txt"

# ===== 3. Hardware & Dtype =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float32
)
print(f"Device: {DEVICE} | Dtype: {DTYPE}")

# ===== 4. Basismodell & Prozessor =====
BASE_ID = "google/gemma-3-4b-it"

if DEVICE == "cuda":
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
    )
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        BASE_ID,
        torch_dtype=DTYPE,
        device_map={"": 0},
        quantization_config=qconf,
    )
else:
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        BASE_ID, torch_dtype=DTYPE, device_map="auto"
    )
base_model.eval()

processor = AutoProcessor.from_pretrained(BASE_ID, use_fast=True)


# ===== 5. PathVQA-Validation laden =====
dataset = load_dataset("flaviagiammarino/path-vqa", split="validation")
# Subset anhand externer Indizes auswählen
if not INDICES_FILE.exists():
    raise FileNotFoundError(f"Indices-Datei nicht gefunden: {INDICES_FILE}")
with INDICES_FILE.open(encoding="utf-8") as f:
    selected_indices = [int(line.strip()) for line in f if line.strip()]

dataset = dataset.select(selected_indices)
print(f"Anzahl der ausgewählten Validation-Samples: {len(dataset)}")

# ===== 6. CSV vorbereiten =====
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
if OUT_CSV.exists():
    df = pd.read_csv(OUT_CSV, quoting=csv.QUOTE_ALL)
else:
    df = pd.DataFrame(
        columns=[
            "ID",
            "model_dir",
            "question",
            "correct_answer",
            "model_output",
        ]
    )
    df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL)

global_id = int(df["ID"].max()) + 1 if not df.empty else 1

# ===== 7. Checkpoints evaluieren =====
for ckpt_dir in sorted(MODEL_ROOT.glob("checkpoint-*")):
    if not ckpt_dir.is_dir():
        continue

    print(f"\n🔹 Evaluating {ckpt_dir.name}")
    print(f"Vor Adapter:  {base_model.num_parameters():,} Parameter")

    # 7a. LoRA-Adapter laden
    model = PeftModel.from_pretrained(base_model, ckpt_dir)
    model.eval()
    print(f"Nach Adapter: {model.num_parameters():,} Parameter")

    # 7b. Inferenz
    for sample in tqdm(dataset, desc=ckpt_dir.name):
        img = sample["image"]
        question = sample["question"]
        correct_answer = sample["answer"]

        # Chat-Prompt (Gemma-Spezialformat)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a medical pathology expert. "
                            "Answer strictly based on the visual information in the image. "
                            "Use short precise terms without explanations."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            },
        ]

        # Tokenisierung
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=DTYPE)

        prompt_len = inputs["input_ids"].shape[-1]

        # Generierung
        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )
        out_ids = gen[0][prompt_len:]
        answer = processor.decode(out_ids, skip_special_tokens=True).strip()

        # CSV-Eintrag
        df.loc[len(df)] = {
            "ID": global_id,
            "model_dir": ckpt_dir.name,
            "question": question,
            "correct_answer": correct_answer,
            "model_output": answer,
        }
        global_id += 1

    # Ergebnisse speichern
    df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"Ergebnisse gespeichert ({OUT_CSV.name})")

print("\nAlle Modelle fertig evaluiert.")