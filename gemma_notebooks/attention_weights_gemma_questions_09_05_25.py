#!/usr/bin/env python3

import os
import csv
from pathlib import Path
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login


def resize_with_aspect_ratio(img, max_size: int = 384):
    """
    Skaliert das Bild so, dass die größte Seite 'max_size' Pixel beträgt
    und das Seitenverhältnis erhalten bleibt.
    """
    w, h = img.size
    scale = max_size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)))


def main():
    # Optional: Login mit HF_TOKEN Umgebungsvariable
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)
    # Device und Datentyp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float

    # Path-VQA Validierungs-Dataset laden
    print("Lade Validation-Dataset...")
    dataset = load_dataset("flaviagiammarino/path-vqa", split="validation")

    # Model und Processor
    model_id = "google/gemma-3-4b-it"
    print(f"Lade Modell {model_id} mit dtype={dtype} auf {device}...")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="eager"
    ).eval().to(device)

    # Ausgabepfad
    out_dir = Path("attention_results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "validation_attention.csv"

    # CSV schreiben
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "token", "score"]);

        total = len(dataset)
        for idx, sample in enumerate(dataset):
            question = sample["question"]
            image = resize_with_aspect_ratio(sample["image"])

            # Eingabe-Template
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]}
            ]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            ).to(device)

            # Forward-Pass mit Attention
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True
                )

            # Aufmerksamkeit der letzten Schicht
            last_attn = outputs.attentions[-1][0].to(torch.float32)  # (heads, seq_len, seq_len)
            attn_avg = last_attn.mean(dim=0).cpu().numpy()           # (seq_len, seq_len)

            # Token-IDs und -Strings
            input_ids = inputs["input_ids"][0].cpu().tolist()
            tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

            # Frage-Tokens isolieren
            q_tokinfo = processor.tokenizer(
                question,
                add_special_tokens=False,
                return_attention_mask=False
            )
            q_ids = q_tokinfo["input_ids"]
            q_tokens = processor.tokenizer.convert_ids_to_tokens(q_ids)
            q_set = {t.lstrip("Ġ") for t in q_tokens}

            # Scores pro Token (Durchschnitt über "von allen Köpfen"")
            token_scores = attn_avg.mean(axis=0)

            # Schreiben aller Frage-Token-Zeilen
            for tok, score in zip(tokens, token_scores):
                cleaned = tok.lstrip("Ġ")
                if cleaned in q_set:
                    writer.writerow([idx, question, cleaned, float(score)])

            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                print(f"Verarbeitet {idx + 1}/{total} Samples")

    print(f"Fertig! Ergebnisse gespeichert in: {csv_path}")


if __name__ == "__main__":
    main()

