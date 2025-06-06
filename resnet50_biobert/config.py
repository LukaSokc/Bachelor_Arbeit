"""
src/config.py

Globale Konfigurationseinstellungen für dein Projekt:
- Pfade
- Hyperparameter
- Trainings-Settings
"""

import os
from dataclasses import dataclass

@dataclass
class Config:
    # Pfade
    DATA_DIR: str = os.path.join("data")
    PROCESSED_DATA_DIR: str = os.path.join("data")
    LOG_DIR: str = os.path.join("logs", "resnet50")
    MODEL_DIR: str = os.path.join("models", "resnet50")
    DOCS_DIR: str = os.path.join("docs", "baseline")

    # Neuer Arrow-Pfad
    TRAIN_ARROW_DIR: str = os.path.join("data", "train")
    VAL_ARROW_DIR: str = os.path.join("data", "validation")
    TEST_ARROW_DIR: str = os.path.join("data", "test")

    # Trainings-Hyperparameter – angepasst an die im Paper angegebenen Werte
    EPOCHS: int = 20
    BATCH_SIZE: int = 16
    LR: float = 4.734917e-05
    SEED: int = 42

    # Modell-Hyperparameter
    MAX_QUESTION_LEN: int = 49
    NHEAD: int = 16
    NUM_ENCODER_LAYERS: int = 2  # Wird über den benutzerdefinierten Encoder abgehandelt
    NUM_DECODER_LAYERS: int = 2
    DIM_FEEDFORWARD: int = 4096
    DROPOUT: float = 0.24128449788146303

    # BioBERT
    BIOBERT_MODEL: str = "dmis-lab/biobert-v1.1"

    # Decoder-Vokabular
    VOCAB_SIZE: int = 30522  # Entsprechend BERT-Base oder eigenem Vokabular