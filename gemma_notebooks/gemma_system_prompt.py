from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm
import torch
import csv

# Modell setup
model_id = "merged_model_batchsize_2"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map={"":0},torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained(model_id, use_fast = True)

# Dtype basierend von Hardware auswählen
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float
print("🖥️ Torch device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# PathVQA Daten holen
project_root = Path.cwd().parent
data_path = project_root / "data" / "validation" 
dataset = load_from_disk(str(data_path))

# Output CSV erstellen
output_file = "../docs/gemma/fine_tuned_batchsize_2.csv"
fieldnames = ["ID", "question", "correct_answer", "model_output"]

with open(output_file, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Prozessierung von Frage-Antwort
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        image = sample["image"]
        question = sample["question"]
        ground_truth = sample["answer"]
        qid = sample.get("id", idx + 1) 

        try:
            # Chat input vorbereiten
            messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a medical expert. Your task is to answer medical questions based solely on the visual information in the provided image. Focus only on what is visible in the image — do not rely on prior medical knowledge, assumptions, or external information. Your responses should be short, factual, and medically precise, using appropriate terminology. Do not include any explanations, reasoning, or additional text. Use a consistent format, without punctuation, and avoid capitalisation unless medically required. Only return the exact answer."
}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }
]

            # Tokenisierung mit Chat Template
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=dtype)

            input_len = inputs["input_ids"].shape[-1]

            # Antwort generieren
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
                output = output[0][input_len:]
            llm_answer = processor.decode(output, skip_special_tokens=True).strip()

            # Ins CSV schreiben
            writer.writerow({
                "ID": qid,
                "question": question,
                "correct_answer": ground_truth,
                "model_output": llm_answer
            })

        except Exception as e:
            print(f"Error on sample {qid}: {e}")
            continue

print("All results saved!")
