# Reinforcement Learning: Vergleich von Q-Learning und SARSA

Dieses Repository enthält die Implementierung und experimentelle Auswertung der in der Masterarbeit durchgeführten Untersuchung zum Vergleich der Reinforcement-Learning-Algorithmen **Q-Learning** und **SARSA**.

## Ziel der Arbeit

Ziel ist der systematische Vergleich der beiden tabellarischen, modellfreien Reinforcement-Learning-Verfahren:

- Q-Learning (Off-Policy)
- SARSA (On-Policy)

Die Analyse erfolgt in zwei diskreten Benchmark-Umgebungen:

- `FrozenLake-v1`
- `Taxi-v3`

Untersucht werden insbesondere:

- Konvergenzverhalten
- Stabilität über mehrere Seeds
- Erfolgsrate
- durchschnittlicher Return
- Episodenlänge

---

## Projektstruktur

```
rl_qlearning_sarsa/
│
├── src/
│   ├── qlearning.py
│   ├── sarsa.py
│   ├── runner.py
│   ├── policy.py
│   ├── eval.py
│   └── utils.py
│
├── experiments/
│   ├── run_all.py
│   ├── test_qlearning.py
│   ├── analyze_results.py
│   ├── results/        # CSV-Ergebnisse (wird bei Experimenten gefüllt)
│   └── figures/        # (optional) generierte Plots
│
├── requirements.txt
└── README.md
```

### Ordnerbeschreibung

- `src/`  
  Enthält die Implementierung der Algorithmen und die Trainings-/Evaluationslogik.

- `experiments/`  
  Enthält Skripte zur Durchführung und Auswertung der Experimente.

- `experiments/results/`  
  Hier werden automatisch alle CSV-Dateien mit den Ergebnissen gespeichert (je Kombination aus Algorithmus, Umgebung und Seed).

---

## Installation

### 1. Repository klonen

```bash
git clone <repository-url>
cd Masterarbeit-code
```

### 2. Virtuelle Umgebung erstellen

```bash
python -m venv .venv
```

### 3. Virtuelle Umgebung aktivieren

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
# oder (ältere shells)
.venv\Scripts\activate
```

Mac / Linux:

```bash
source .venv/bin/activate
```

### 4. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

Wichtige Abhängigkeiten (Auszug):

- `numpy`
- `pandas`
- `gymnasium`
- `matplotlib`

---

## Experimente ausführen

### Einzelnes Experiment testen

Aus dem Projekt-Root:

```powershell
python .\rl_qlearning_sarsa\experiments\test_qlearning.py
```

> Hinweis: Die Skripte fügen bei Bedarf automatisch das Projekt-Root zu `sys.path` hinzu, damit `from src.runner import ...` auch funktioniert, wenn du das Skript direkt startest.

### Alle Experimente ausführen

```powershell
cd .\rl_qlearning_sarsa
python .\experiments\run_all.py
```

Dies erzeugt für jede Kombination aus Algorithmus (Q-Learning / SARSA), Umgebung (FrozenLake / Taxi) und Seed eine Ergebnisdatei im Ordner:

```
rl_qlearning_sarsa/results/
```

Jede Ergebnisgruppe enthält pro Run:
- `*_train.csv` (Episode-Log)
- `*_summary.csv` (zusammenfassende Metriken)

---

## Ergebnisanalyse

Zur Aggregation und Analyse der Ergebnisse:

```bash
python .\rl_qlearning_sarsa\experiments\analyze_results.py
```

Das Skript:
- lädt alle `*_summary.csv` Dateien
- berechnet Mittelwerte und Standardabweichungen über Seeds
- speichert eine aggregierte Tabelle (`aggregated_summary.csv`)
- erstellt Plots in `rl_qlearning_sarsa/figures/`

---

## Reproduzierbarkeit

Alle Experimente sind reproduzierbar durch:

- feste Random Seeds (in den Experiment-Skripten gesetzt)
- dokumentierte Hyperparameter
- Verwendung von `gymnasium` mit versioniertem API

### Standard-Hyperparameter (im Projekt)

- Lernrate (α): `0.1`
- Diskontierungsfaktor (γ): `0.99`
- ε-start: `1.0`
- ε-end: `0.05` (oder `0.1` in Testskripten)
- ε-decay: `0.995` (bzw. `0.99` für kurze Tests)

---

## Bewertungsmetriken

Für jede Kombination werden folgende Kennzahlen berechnet und in `*_summary.csv` abgelegt:

- `eval_mean_return`
- `eval_std_return`
- `eval_mean_length`
- `eval_success_rate`

Diese Metriken bilden die Grundlage der Analyse im Kapitel 10 der Masterarbeit.

---

## Wissenschaftlicher Kontext

Die Implementierung orientiert sich an den klassischen tabellarischen Verfahren (siehe insbesondere):

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).

---

## Lizenz

Dieses Repository wurde im Rahmen einer wissenschaftlichen Abschlussarbeit erstellt. Falls du es teilen möchtest, verwende bitte eine passende Lizenz (z. B. MIT) und dokumentiere die Nutzung in der Arbeit.

---

Wenn du willst, kann ich die README noch um eine example-Output-Sektion ergänzen, oder die Installationsschritte für Conda ergänzen.
