# Kompakte Multimodale Sprachmodelle in der Pathologie-VQA

**Bachelorarbeit an der Zürcher Hochschule für Angewandte Wissenschaften (ZHAW)**  
Autoren: Arbnor Ziberi, Luka Sokcevic  
Betreuerin: Prof. Dr. Jasmina Bogojeska

---

## Projektbeschreibung

Dieses Projekt untersucht den Einsatz kompakter multimodaler Sprachmodelle (MLLMs) für Visual Question Answering (VQA) in der Pathologie.

Im Zentrum stehen ein Leistungsvergleich zwischen Gemma 3 4B und Qwen 2.5VL 3B auf dem PathVQA-Datensatz, sowie Erklärbarkeitsansätze zur Modelltransparenz.  
Alle Hintergründe, Motivation und Ziele sind im PDF der Bachelorarbeit detailliert beschrieben.

---

## Installation

Für das lokale Setup benötigst du eine GPU (mindestens T4 für Inferenz, LS40 für Finetuning empfohlen).

**Schritt-für-Schritt-Anleitung:**

1. Repository clonen:  
   ```bash
   git clone https://github.com/LukaSokc/Bachelor_Arbeit.git
   cd Bachelor_Arbeit
2. Abhängigkeiten installieren (je nach Modell):
   - Für Gemma (Fine-Tuning & Inferenz):
    pip install -r requirements_gemma.txt
    - Für Qwen (Fine-Tuning & Inferenz):
      pip install -r requirements_qwen.txt


Wichtig: Die beiden requirements-Dateien sind nicht kompatibel. Immer nur eine installieren, je nachdem, mit welchem Modell gearbeitet wird.

3. Mit ausreichend GPU-Ressourcen können die MLLMs wie in der Arbeit beschrieben fine-getuned werden.

## Nutzung
- Trainieren/Fine-Tuning:
Das Fine-Tuning der Modelle wurde auf einer LS40 GPU durchgeführt (siehe Kapitel Hardware im PDF).

- Inferenz:
Für Inferenz genügt eine T4 GPU (wie im Projekt getestet).

## Abhängigkeiten
Alle benötigten Libraries sind in den jeweiligen requirements_*.txt aufgelistet.
Es gibt keine API-Keys oder weitere versteckte Konfigurationen.

## Beitragende
Bei Fragen zum Projekt oder bei Interesse an einer Zusammenarbeit:

- Arbnor Ziberi: ziberarb@students.zhaw.ch

- Luka Sokcevic: sokceluk@students.zhaw.ch


Detaillierte Informationen zu Motivation, wissenschaftlichem Hintergrund, Methoden, Daten, Ergebnissen und Erklärbarkeit der Modelle findest du direkt in der Bachelorarbeit im PDF-Format.
