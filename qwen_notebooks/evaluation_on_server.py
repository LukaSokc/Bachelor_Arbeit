import csv
import os
import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from pathlib import Path
from peft import PeftModel

# Parameter
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Vortrainiertes Modell laden (mit 4-Bit Quantisierung falls CUDA verfügbar)
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=True
    )
else:
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        use_cache=True
    )

# 2. Prozessor laden
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

# 3. Adapter laden (Pfad relativ zum Skript, egal von wo gestartet)
script_dir = Path(__file__).resolve().parent
adapter_path = script_dir / "output_bs2"

print(f"Before adapter parameters: {base_model.num_parameters()}")
model = PeftModel.from_pretrained(base_model, str(adapter_path))
print(f"After adapter parameters: {model.num_parameters()}")

# 4. Validation-Datensatz laden
dataset = load_dataset("flaviagiammarino/path-vqa", split="validation")

# 5. Ergebnis-CSV vorbereiten
csv_file = script_dir / "results_finetuned_qwen_bs2_val_with_system_message_teeeeessttttt_new.csv"

# Falls Datei existiert: Start-ID ermitteln
if csv_file.exists():
    try:
        existing_df = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL)
        if "ID" in existing_df.columns:
            last_id = int(existing_df["ID"].max())
        else:
            last_id = len(existing_df)
    except Exception:
        last_id = 0
else:
    last_id = 0
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "question", "correct_answer", "model_output"], quoting=csv.QUOTE_ALL)
        writer.writeheader()

# 6. System Message definieren
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

# 7. Inferenz starten ab Index 5875
start_index = 0
for idx in range(start_index, len(dataset)):
    sample = dataset[idx]
    image = sample["image"]
    question = sample["question"]
    correct_answer = sample["answer"]

    # Nachrichten zusammenstellen: System + User
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_message}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    # Prompt-Text und Vision-Eingaben vorbereiten
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Antwort generieren
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Prompt-Anteil entfernen
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    model_output = output_text[0]

    # Ergebnis speichern
    last_id += 1
    row = {
        "ID": last_id,
        "question": question,
        "correct_answer": correct_answer,
        "model_output": model_output
    }

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "question", "correct_answer", "model_output"], quoting=csv.QUOTE_ALL)
        writer.writerow(row)

    print(f"Sample {idx} verarbeitet und in CSV abgespeichert.")


