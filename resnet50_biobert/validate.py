import os
import csv
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from resnet50_biobert.config import Config
from resnet50_biobert.dataset import PathVQADataset
from resnet50_biobert.model_cnn import ImageFeatureExtractor
from resnet50_biobert.model_text import BioBERTBiLSTM
from resnet50_biobert.model_transformer import TraPVQA

def main():
    # Konfiguration und Gerät initialisieren
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(cfg.BIOBERT_MODEL)
    
    # Definiere Start- und End-Token anhand des Tokenizers:
    start_token_id = tokenizer.cls_token_id  # [CLS]-Token, z. B. 101
    eos_token_id = tokenizer.sep_token_id      # [SEP]-Token, z. B. 102

    # Validierungs-Datensatz laden
    val_ds = PathVQADataset(
        arrow_dir=cfg.VAL_ARROW_DIR,
        image_dir=cfg.DATA_DIR,
        tokenizer=tokenizer,
        max_length=cfg.MAX_QUESTION_LEN
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Modelle initialisieren – gleiche Architektur wie im Training
    text_encoder = BioBERTBiLSTM(
        biobert_model=cfg.BIOBERT_MODEL,
        lstm_hidden=256,
        dropout=cfg.DROPOUT,
        max_len=cfg.MAX_QUESTION_LEN
    )
    image_encoder = ImageFeatureExtractor(
        dropout=cfg.DROPOUT,
        out_channels=512
    )
    model = TraPVQA(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        vocab_size=cfg.VOCAB_SIZE,
        nhead=cfg.NHEAD,
        num_encoder_layers=cfg.NUM_ENCODER_LAYERS,
        num_decoder_layers=cfg.NUM_DECODER_LAYERS,
        dim_feedforward=cfg.DIM_FEEDFORWARD,
        dropout=cfg.DROPOUT,
        start_token_id=start_token_id,
        eos_token_id=eos_token_id
    ).to(device)

    # Lade den gespeicherten Checkpoint (Pfad ggf. anpassen)
    checkpoint_path = os.path.join(cfg.MODEL_DIR, "resnet50_biobert_finetuned_2025-05-10_15-12-59.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Checkpoint {checkpoint_path} wurde geladen.")
    else:
        print(f"Checkpoint {checkpoint_path} wurde nicht gefunden. Bitte überprüfe den Pfad.")
        return

    model.eval()
    
    # Ergebnisse sammeln in einer Liste (jede Zeile: ID, question, correct_answer, model_output)
    results = []
    current_id = 1
    
    with torch.no_grad():
        total_samples = len(val_loader)
        for i, batch in enumerate(val_loader, start=1):
            # Batch-Daten laden
            images = batch["image"].to(device)            # (1, 3, 224, 224)
            input_ids = batch["input_ids"].to(device)       # (1, lQ)
            attention_mask = batch["attention_mask"].to(device)  # (1, lQ)
            gold_answer = batch["answer_text"][0]           # Goldstandard als String

            # Inferenz (ohne Teacher Forcing)
            outputs = model(images, input_ids, attention_mask)
            # outputs hat Dimension (B, generierte Sequenzlänge) mit B=1
            pred_tokens = outputs[0].tolist()
            model_output = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            
            # Frage als Text aus den input_ids
            question_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # Zeile hinzufügen
            results.append([current_id, question_text, gold_answer, model_output])
            current_id += 1

            print(f"Verarbeitet: {i}/{total_samples} Beispiele")
    
    # Schreibe die Ergebnisse in eine CSV-Datei mit alle Feldern in Anführungszeichen
    csv_file = os.path.join("docs", "validation_results_resnet50_biobert_finetuned.csv")
    os.makedirs("docs", exist_ok=True)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["ID", "question", "correct_answer", "model_output"])
        writer.writerows(results)
    print(f"Validation-Ergebnisse wurden in {csv_file} gespeichert.")

if __name__ == "__main__":
    main()
