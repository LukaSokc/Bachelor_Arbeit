import csv
import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk
from pathlib import Path
from peft import PeftModel
from PIL import Image
from tqdm import tqdm



# === 0. Settings ===
MODEL_ID    = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR  = Path(__file__).resolve().parent
CHECKPOINTS = SCRIPT_DIR.parent / "models" / "qwen" / "output_bs2"   # hier liegen deine checkpoint‑Ordner
CSV_PATH    = SCRIPT_DIR.parent / "docs" / "qwen" / "results_all_checkpoints_bs2_test.csv"
INDICES_FILE = SCRIPT_DIR.parent / "data" / "validation_subset_indices.txt"

# === 1. Prozessor laden (einmal) ===
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

# === 2. Validation‑Set laden ===
project_root = Path.cwd().parent
data_path_val = project_root / "data" / "validation"
dataset = load_from_disk(str(data_path_val))

# === 3. Subset anhand externer Indizes auswählen ===
if not INDICES_FILE.exists():
    raise FileNotFoundError(f"Indices‑Datei nicht gefunden: {INDICES_FILE}")
with INDICES_FILE.open("r", encoding="utf-8") as f:
    selected_indices = [int(line.strip()) for line in f if line.strip()]
dataset = dataset.select(selected_indices)
print(f"Anzahl der ausgewählten Validation‑Samples: {len(dataset)}")

# === 4. CSV initialisieren ===
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH, quoting=csv.QUOTE_ALL)
else:
    df = pd.DataFrame(columns=["ID", "model", "question", "correct_answer", "model_output"])
    df.to_csv(CSV_PATH, index=False, quoting=csv.QUOTE_ALL)

global_id = int(df["ID"].max()) + 1 if not df.empty else 1

# === 5. Checkpoints finden & evaluieren ===
print("Looking for checkpoints in", CHECKPOINTS)
if not CHECKPOINTS.exists():
    raise FileNotFoundError(f"Checkpoint‑Ordner nicht gefunden: {CHECKPOINTS}")

for ckpt in sorted(CHECKPOINTS.iterdir()):
    if not ckpt.is_dir() or "checkpoint" not in ckpt.name:
        continue
    ckpt_name = ckpt.name
    print(f"\n--- Testing model {ckpt_name} ---")

    # === 6. Basis‑Modell für diesen Checkpoint frisch laden ===
    if DEVICE == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb,
            device_map="auto",
            use_cache=True
        )
    else:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            use_cache=True
        )

    # Zeige Parameterzahl vor Adapter
    print(f"Before adapter parameters: {base_model.num_parameters():,}")

    # === 7. Adapter‑Modell laden & auf Device schieben ===
    model = PeftModel.from_pretrained(base_model, str(ckpt)).to(DEVICE)

    # Zeige Parameterzahl nach Adapter
    print(f"After adapter parameters : {model.num_parameters():,}")

    model.eval()

    # === 8. Inferenz über das selektierte Subset ===
    for sample in tqdm(dataset, desc=ckpt_name):
        # Bild laden
        img = sample["image"]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        # Nachrichten zusammenstellen
        system_message = (
            "You are a medical pathology expert. Your task is to answer medical questions "
            "based solely on the visual information in the provided pathology image. "
            "Focus only on what is visible in the image — do not rely on prior medical knowledge, "
            "assumptions, or external information. Your responses should be short, factual, "
            "and medically precise, using appropriate terminology. "
            "Do not include any explanations, reasoning, or additional text. "
            "Use a consistent format, without punctuation, and avoid capitalisation unless medically required. "
            "Only return the exact answer."
        )
        messages = [
            {"role": "system", "content":[{"type":"text","text":system_message}]},
            {"role": "user",   "content":[{"type":"image","image":img},
                                           {"type":"text","text":sample["question"]}]}
        ]

        # Prompt & Vision‑Inputs
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        imgs, vids = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=imgs,
            videos=vids,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        # Antwort generieren
        gen = model.generate(**inputs, max_new_tokens=128)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen)]
        output = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        # Ergebnis speichern mit .loc
        df.loc[len(df)] = {
            "ID":             global_id,
            "model":          ckpt_name,
            "question":       sample["question"],
            "correct_answer": sample["answer"],
            "model_output":   output
        }
        global_id += 1

    # === 9. Zwischenspeichern ===
    df.to_csv(CSV_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f" → Ergebnisse für {ckpt_name} gespeichert in {CSV_PATH}")

print("\n✅ Fertig! Alle Modelle evaluiert.")  
