# 🏥 7la2 els7a

A medical question-answering chatbot powered by a fine-tuned **BioT5** model.

## Features

- **Fine-tuned** on medical QA dataset
- **Web Interface** (Flask)
- **CLI Chat** option

## Quick Start

### 1. Install Dependencies
```bash
pip install tensorflow transformers sentencepiece tf-keras flask
```

### 2. Run Web App
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

### 3. Run CLI Chat
```bash
python chat_biot5.py
```

## Project Structure

```
7la2 els7a/
├── app.py                  # Flask web server
├── chat_biot5.py           # CLI chat script
├── templates/
│   └── index.html          # Web UI
└── transformer/
    └── final t5 model with loss/   # Fine-tuned model
        ├── config.json
        ├── tf_model.h5
        ├── spiece.model
        └── tokenizer_config.json
```

## Model Details

| Property | Value |
|----------|-------|
| Base Model | `QizhiPei/biot5-base` |
| Framework | TensorFlow |
| Training | Custom loop, Float32, Single GPU |
| Max Length | 256 tokens |

## Usage Example

**Input**: "What are the symptoms of diabetes?"

**Output**: "Common symptoms of diabetes include increased thirst, frequent urination, fatigue, blurred vision, and slow-healing wounds..."

## License

For educational purposes.
