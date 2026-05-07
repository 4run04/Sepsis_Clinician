# Sepsis AI Clinician

Sepsis AI Clinician is a modular, Reinforcement Learning (RL) based recommendation engine designed to suggest optimal treatment policies (IV Fluids and Vasopressors) for patients with Sepsis.

This project implements a **Highlight Dueling Double Deep Q-Network (Highlight-DDQN)** to learn clinically actionable policies from Electronic Health Record (EHR) data. It is capable of working with real-world clinical datasets, such as MIMIC-III, or generating synthetic clinical profiles for robust policy testing.

## Table of Contents
- [Architecture](#architecture)
- [State and Action Spaces](#state-and-action-spaces)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Project Structure](#project-structure)

## Architecture

The project has been refactored into a structured, modular pipeline (v2.0) for better maintainability and clarity:

- **Data Ingestion & Preprocessing**: Handles importing EHR data, missing value imputation (e.g., MICE), and trajectory formatting.
- **Model (Highlight-DDQN)**: A PyTorch-based neural network combining Dueling architecture, Double Q-learning, and a specialized highlight mechanism for efficient exploration.
- **Agent**: Manages experience replay, policy execution, and neural network training steps.
- **Evaluation Engine**: Calculates survival probabilities, action distribution metrics, Q-values, and entropy.

## State and Action Spaces

### State Space
The environment models a patient's state using **48 continuous clinical features**, derived from vitals, lab values, and demographic data. Key features include:
- Vitals: Heart Rate, Blood Pressure (sys, dia, mean), Temperature, SpO2, Respiratory Rate
- Labs: Lactate, Creatinine, BUN, Glucose, Potassium, Bilirubin
- Severity Scores: SOFA, Shock Index

### Action Space
The agent outputs a discrete action mapped to a **5x5 grid (25 total actions)** representing the concurrent administration of:
- **IV Fluids (5 levels)**
- **Vasopressors (5 levels)**

### Rewards
The model trains on sparse binary outcomes at the end of the patient trajectory:
- **Survival**: +1.0
- **Mortality**: -1.0

## Installation

Ensure you have Python 3.8+ installed.

1. Clone or navigate to the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include: `torch`, `numpy`, `pandas`, `scikit-learn`, `datasets`, and `streamlit`.

## Usage

### Training

To train the AI Clinician from scratch, use the `main.py` script. The script handles data ingestion (pulling a MIMIC-III derived dataset from Hugging Face if no local copy exists), preprocessing, training multiple RL sessions, and saving the best model.

```bash
python main.py
```

- Training hyperparameters and configurations can be adjusted in `config.py`.
- The best performing model weights will be saved to `checkpoints/best_model.pt`.
- Results and metrics are logged in the `results/` directory.

### Inference

Once trained, use the `run_inference.py` script to generate treatment recommendations. The script supports evaluating custom EHR CSV datasets or evaluating internally generated diverse clinical profiles (personas).

**Run default heuristic profile inference:**
```bash
python run_inference.py --num-samples 10
```

**Run inference on an external EHR dataset:**
```bash
python run_inference.py --test-data path/to/your/data.csv --output results/predictions.csv
```

## Project Structure

```text
Sepsis_AI/
├── main.py               # Main orchestration script for training and evaluation
├── run_inference.py      # Inference engine for batch predictions and persona testing
├── config.py             # Global constants, hyperparams, and feature definitions
├── agent.py              # RL Agent and Replay Buffer implementation
├── model.py              # PyTorch Network architectures (Highlight-DDQN)
├── data_utils.py         # Data preprocessing, imputation, and synthetic generation
├── evaluate.py           # Functions for model evaluation and feature importance
├── requirements.txt      # Project dependencies
├── checkpoints/          # Directory where trained model weights (.pt) are saved
├── data/                 # Directory for raw and preprocessed datasets (.csv, .pkl)
└── results/              # Directory for evaluation outputs and metrics
```

## Disclaimer

**For Research Purposes Only.** This model is an experimental AI research tool. It is not approved for clinical use and should not be used to make actual medical decisions.
