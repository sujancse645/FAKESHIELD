# FakeShield: Evidence-Grounded Multilingual Multimodal News Verification System

FakeShield is a machine-learning-based news verification system designed to analyze potentially misleading or false information appearing as text, newspaper clippings, screenshots, and URLs.

Unlike traditional fake-news classifiers that attempt to predict whether text "looks fake", FakeShield follows an evidence-grounded claim verification approach.

## Directory Structure

```
FakeShield/
│
├── PROJECT_SPEC.md       # Complete project specification and verification gates
├── data/                 # Datasets, golden set, raw & processed items
├── src/                  # Core source code (OCR, translation, claim extraction, NLI, retrieval, UI)
├── eval/                 # Evaluation scripts, baseline models, metrics
├── configs/              # System and source reliability configurations
├── notebooks/            # Exploratory analysis and experiments
└── README.md             # Project readme
```

Refer to [PROJECT_SPEC.md](PROJECT_SPEC.md) for detailed architecture, methodology, and verification phases.
