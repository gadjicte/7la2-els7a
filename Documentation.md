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