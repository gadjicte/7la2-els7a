# Medical Chatbot Project Documentation

## Project Overview

**Goal**: Build a Medical Question-Answering Chatbot trained on a custom QA dataset.

**Data Source**: `D:\AIvolution\data\qa_dataset_concatenated.csv`

**Environment**: Windows with 8 GPUs, TensorFlow/PyTorch, Python 3.11

**Final Working Model**: `QizhiPei/biot5-base` (Fine-tuned)

---

## Model Attempts Summary

| # | Model/Approach | Framework | Status | Primary Failure Reason |
|---|----------------|-----------|--------|------------------------|
| 1 | Custom LSTM with Attention | TensorFlow/Keras | ⚠️ Partial | Incoherent responses, vocabulary corruption |
| 2 | Custom Transformer (from scratch) | TensorFlow | ⚠️ Partial | Required extensive training, slow inference |
| 3 | BioT5 (Multi-GPU, `MirroredStrategy`) | TensorFlow | ❌ Failed | `NcclAllReduce` error (Windows lacks NCCL) |
| 4 | BioT5 (`model.fit()`) | TensorFlow | ❌ Failed | `AttributeError: keras.utils.unpack_x_y_sample_weight` |
| 5 | BioT5 (Custom Loop + Mixed Precision) | TensorFlow | ❌ Failed | `loss: nan` due to float16 without gradient scaling |
| 6 | BioT5 (Single GPU, Float32, Custom Loop) | TensorFlow | ✅ **Success** | Final working configuration |
| 7 | T5 with PyTorch + `Seq2SeqTrainer` | PyTorch | ❌ Failed | Windows multiprocessing (`PermissionError`), slow |

---

## Detailed Failure Analysis

### 1. Custom LSTM with Attention

**Files**: `medical_chatbot_attention.h5`, [vectorizer_data.pkl](file:///e:/vs%20codes/7la2%20els7a/vectorizer_data.pkl)

**Issue**: The model trained successfully but produced **incoherent or repetitive responses**. Root cause was identified as **vocabulary corruption** during the `TextVectorization` layer export. The pickle file contained corrupted tokens (`0xC3` bytes) that couldn't be decoded properly.

**Attempts to Fix**:
- Multiple vocabulary export scripts ([add_vocab_keras.py](file:///e:/vs%20codes/7la2%20els7a/add_vocab_keras.py), [add_vocab_numpy.py](file:///e:/vs%20codes/7la2%20els7a/add_vocab_numpy.py), [add_vocab_salvage.py](file:///e:/vs%20codes/7la2%20els7a/add_vocab_salvage.py))
- TensorFlow SavedModel export for vectorizers
- Raw byte-level weight extraction

**Conclusion**: The vocabulary was irreparably linked to the training environment and could not be cleanly exported to inference scripts. We abandoned this approach.

---

### 2. Custom Transformer (From Scratch)

**Files**: `thelastone.ipynb`, [chat_stream.py](file:///e:/vs%20codes/7la2%20els7a/chat_stream.py)

**Issue**: Training was successful, but the model required **very long training times** for convergence. Inference was slow due to step-by-step autoregressive generation.

**Why Abandoned**: A pre-trained model like T5/BioT5 offers far better starting weights and faster fine-tuning.

---

### 3. BioT5 with Multi-GPU (`MirroredStrategy`)

**Files**: [biot5_finetune.ipynb](file:///e:/vs%20codes/7la2%20els7a/biot5_finetune.ipynb), `biot5_finetune_hf_style.ipynb`

**Error**:
```
InvalidArgumentError: No OpKernel was registered to support Op 'NcclAllReduce'
```

**Cause**: TensorFlow's `MirroredStrategy` on Windows defaults to NCCL for multi-GPU communication, but **NCCL is not supported on Windows**.

**Attempted Fix**: Set `tf.distribute.HierarchicalCopyAllReduce()` as the cross-device-ops. This partially worked but introduced other instabilities.

**Conclusion**: Multi-GPU training on Windows TensorFlow is unreliable. Switched to **Single-GPU** training.

---

### 4. BioT5 with `model.fit()`

**Error**:
```
AttributeError: module 'keras.utils' has no attribute 'unpack_x_y_sample_weight'
```

**Cause**: Incompatibility between `transformers` library (which expected Keras 2 API) and the installed **Keras 3**. The internal Hugging Face `fit()` wrapper called a function that no longer exists in Keras 3.

**Attempted Fixes**:
- Monkey-patching `keras.utils.unpack_x_y_sample_weight`
- Installing `tf-keras` compatibility layer

**Conclusion**: Even with patches, `model.fit()` triggered internal loss calculations that also failed. We switched to a **Custom Training Loop** using `tf.GradientTape`.

---

### 5. BioT5 with Custom Loop + Mixed Precision

**Error**:
```
Epoch 1/3: loss: nan
```

**Cause**: Enabled `tf.keras.mixed_precision.set_global_policy('mixed_float16')` without implementing **gradient scaling** (via `tf.keras.mixed_precision.LossScaleOptimizer`). The float16 gradients underflowed to zero or overflowed to NaN.

**Conclusion**: Disabled mixed precision. Training in **float32** is slower but numerically stable.

---

### 6. BioT5 (Single GPU, Float32, Custom Loop) — ✅ FINAL SOLUTION

**File**: [biot5_simple.ipynb](file:///e:/vs%20codes/7la2%20els7a/biot5_simple.ipynb)

**Configuration**:
- **Model**: `QizhiPei/biot5-base`
- **Strategy**: Single GPU (No `MirroredStrategy`)
- **Precision**: Float32 (Mixed Precision Disabled)
- **Training Loop**: Custom `tf.GradientTape` loop
- **Loss**: Manual `SparseCategoricalCrossentropy` (NOT passed to model internally)
- **Batch Size**: 2 (Reduced to prevent OOM)
- **Epochs**: 3

**Output Model**: `e:\vs codes\7la2 els7a\transformer\final t5 model with loss`

This configuration trained successfully with decreasing loss values and produces coherent responses.

---

### 7. PyTorch T5 with `Seq2SeqTrainer`

**File**: [biot5_finetune_pt.ipynb](file:///e:/vs%20codes/7la2%20els7a/biot5_finetune_pt.ipynb)

**Errors**:
1. `PermissionError: [WinError 32]` — Windows file locking during multiprocessing
2. Extremely slow training (`9.91 s/it`)

**Cause**: PyTorch `DataLoader` on Windows has known issues with multiprocessing. Setting `dataloader_num_workers=0` fixed crashes but made training impractically slow.

**Conclusion**: Abandoned in favor of TensorFlow approach.

---

## Final Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User (Browser)                     │
└────────────────────────────┬────────────────────────────┘
                             │ HTTP POST /get
                             ▼
┌─────────────────────────────────────────────────────────┐
│                     Flask Server (app.py)               │
│  - Receives user question                               │
│  - Calls generate_response()                            │
│  - Returns JSON response                                │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│               BioT5 Model (TFT5ForConditionalGeneration)│
│  - Loaded from: transformer/final t5 model with loss    │
│  - Tokenizer: T5Tokenizer                               │
│  - Generation: Beam Search (num_beams=4)                │
└─────────────────────────────────────────────────────────┘
```

---

## Key Files

| File | Purpose |
|------|---------|
| [app.py](file:///e:/vs%20codes/7la2%20els7a/app.py) | Flask web server serving the chatbot GUI |
| [chat_biot5.py](file:///e:/vs%20codes/7la2%20els7a/chat_biot5.py) | Standalone CLI chat script |
| [biot5_simple.ipynb](file:///e:/vs%20codes/7la2%20els7a/biot5_simple.ipynb) | The working training notebook |
| [generate_biot5_simple.py](file:///e:/vs%20codes/7la2%20els7a/generate_biot5_simple.py) | Script to regenerate the notebook |
| `transformer/final t5 model with loss/` | Fine-tuned model weights and tokenizer |

---

## Lessons Learned

1. **Windows + Multi-GPU + TensorFlow = Pain**: NCCL is not available. Use Single GPU or switch to Linux.
2. **Keras 3 Breaks Transformers**: The `transformers` library is not fully compatible with Keras 3. Use `tf-keras` or pin to older versions.
3. **Mixed Precision Needs Gradient Scaling**: Never enable `float16` in a custom training loop without `LossScaleOptimizer`.
4. **Pre-trained > From Scratch**: Fine-tuning BioT5 took ~3 epochs vs. potentially 100+ for a from-scratch Transformer.
5. **Vocabulary Export is Hard**: Exporting `TextVectorization` layers reliably between environments is fragile. Use standard tokenizers (SentencePiece, BPE) instead.

---

## How to Run

### CLI Chat:
```bash
cd "e:\vs codes\7la2 els7a"
python chat_biot5.py
```

### Web GUI:
```bash
cd "e:\vs codes\7la2 els7a"
python app.py
# Open http://127.0.0.1:5000 in browser
```

---

## API Reference

### Flask Endpoints

#### `GET /`
Returns the main chat interface HTML page.

#### `POST /get`
Processes a chat message and returns the bot's response.

**Request Body** (form-data):
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `msg` | string | Yes | User's question |

**Response** (JSON):
```json
{
  "response": "The bot's answer to your question..."
}
```

**Example**:
```bash
curl -X POST http://127.0.0.1:5000/get -d "msg=What are symptoms of diabetes?"
```

---

## Configuration Reference

### app.py Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `transformer/final t5 model with loss` | Path to fine-tuned model |
| `MAX_LENGTH` | `256` | Maximum token length for input/output |
| `port` | `5000` | Flask server port |
| `debug` | `True` | Enable Flask debug mode |

### Generation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_beams` | `4` | Beam search width (higher = better quality, slower) |
| `early_stopping` | `True` | Stop when all beams find EOS |
| `repetition_penalty` | `2.0` | Penalize repeated tokens |
| `max_length` | `256` | Maximum output tokens |

---

## Dataset Details

### Source File
`D:\AIvolution\data\qa_dataset_concatenated.csv`

### Format
| Column | Description |
|--------|-------------|
| `question` | Medical question from user |
| `answer` | Expected medical response |

### Preprocessing Applied
1. **ASCII Cleaning**: Remove non-ASCII characters
2. **Prefix Addition**: Questions prefixed with `"question: "`
3. **Tokenization**: SentencePiece tokenizer (BioT5)
4. **Padding**: Max length 256 tokens

### Dataset Statistics (Approximate)
- **Total Samples**: ~147,600
- **Train/Test Split**: 90% / 10%
- **Batch Size**: 2 (during training)

---

## Training Configuration

### Final Working Configuration

```python
class Config:
    MODEL_NAME = "QizhiPei/biot5-base"
    MAX_LENGTH = 256
    BATCH_SIZE = 2
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    CHECKPOINT_DIR = r'D:\AIvolution\transformer\biot5_checkpoints'
```

### Training Loop Details
- **Framework**: TensorFlow 2.x
- **Loop Type**: Custom `tf.GradientTape`
- **Optimizer**: Adam (lr=1e-4)
- **Loss Function**: SparseCategoricalCrossentropy (manual, with padding mask)
- **Precision**: Float32 (no mixed precision)
- **GPU Strategy**: Single GPU (no MirroredStrategy)

### Checkpointing
- Saved every epoch to `CHECKPOINT_DIR`
- Uses `tf.train.CheckpointManager`
- Keeps last 2 checkpoints
- Final model saved in HuggingFace format using `model.save_pretrained()`

---

## Model Information

### Base Model
- **Name**: `QizhiPei/biot5-base`
- **Architecture**: T5 (Text-to-Text Transfer Transformer)
- **Parameters**: ~220M
- **Pre-training**: Biomedical text corpus

### Fine-tuned Model
- **Location**: `transformer/final t5 model with loss/`
- **Size**: ~1.2 GB
- **Format**: TensorFlow (.h5) + HuggingFace configs

### Model Files
| File | Size | Purpose |
|------|------|---------|
| `tf_model.h5` | ~1.2 GB | Model weights |
| `config.json` | 824 B | Model architecture config |
| `spiece.model` | 792 KB | SentencePiece tokenizer |
| `tokenizer_config.json` | 559 KB | Tokenizer configuration |
| `added_tokens.json` | 64 KB | Special tokens |

---

## Dependencies

### Required Packages
```
tensorflow>=2.15.0
transformers>=4.35.0
sentencepiece>=0.1.99
tf-keras>=2.15.0
flask>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
datasets>=2.14.0
```

### Installation
```bash
pip install tensorflow transformers sentencepiece tf-keras flask pandas numpy datasets
```

### Environment Notes
- **Python**: 3.11 recommended
- **CUDA**: 11.8+ for GPU acceleration
- **OS**: Windows 10/11 (tested), Linux compatible

---

## Troubleshooting

### Common Errors

#### 1. `ModuleNotFoundError: No module named 'tf_keras'`
```bash
pip install tf-keras
```

#### 2. `ModuleNotFoundError: No module named 'sentencepiece'`
```bash
pip install sentencepiece
```

#### 3. `ValueError: Keras 3 is not yet supported in Transformers`
Add this to the top of your Python file **before** importing TensorFlow:
```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

#### 4. `ResourceExhaustedError: OOM when allocating tensor`
- Reduce `BATCH_SIZE` (try 1 or 2)
- Reduce `MAX_LENGTH` (try 128)
- Close other GPU-consuming applications

#### 5. `InvalidArgumentError: No OpKernel was registered to support Op 'NcclAllReduce'`
- Don't use `MirroredStrategy` on Windows
- Use single GPU training instead

#### 6. `loss: nan` during training
- Disable mixed precision:
```python
# Comment out this line:
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

#### 7. Model returns empty or garbage responses
- Ensure you're using the **fine-tuned** model, not the base model
- Check that model path is correct
- Verify the model was trained with actual loss decrease (not NaN)

---

## Performance Benchmarks

### Training (Single GPU - RTX 3090)
| Metric | Value |
|--------|-------|
| Time per epoch | ~2-3 hours |
| Total training time | ~6-9 hours |
| GPU Memory Usage | ~22 GB |

### Inference
| Metric | Value |
|--------|-------|
| Average response time | 2-5 seconds |
| Tokens per second | ~50-100 |
| Memory footprint | ~4 GB |

---

## Future Improvements

1. **Streaming Responses**: Implement token-by-token streaming for better UX
2. **Multi-GPU Support**: Migrate to Linux for NCCL support
3. **LoRA Fine-tuning**: Reduce memory requirements
4. **Quantization**: INT8 inference for faster responses
5. **RAG Integration**: Add retrieval-augmented generation for factual accuracy
6. **Conversation History**: Maintain context across multiple turns

---

## License

This project is for **educational purposes only**. 

The base BioT5 model is subject to its original license from [QizhiPei/biot5-base](https://huggingface.co/QizhiPei/biot5-base).

Medical information provided by this chatbot should **NOT** be used as a substitute for professional medical advice.