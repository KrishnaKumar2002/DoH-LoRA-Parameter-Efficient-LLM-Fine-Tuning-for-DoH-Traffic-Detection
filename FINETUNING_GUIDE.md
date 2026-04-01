# DoH-LoRA Fine-Tuning & GitHub Push Guide

## 📋 Quick Run Guide

### Step 1: Run Fine-Tuning

```bash
python finetune.py
```

This script will:
- ✓ Verify CSV data files exist
- ✓ Install optional dependencies (bitsandbytes for 8-bit quantization)
- ✓ Run the two-stage fine-tuning pipeline
- ✓ Generate comprehensive results:
  - Confusion matrices (PNG)
  - Classification reports (TXT)
  - Metrics summary (CSV + JSON)
  - LoRA adapters (saved models)
  - Predictions on test set (CSV)

**Estimated Time:**
- CPU: 2-4 hours total
- GPU (T4): 30-45 minutes total
- GPU (A100): 5-15 minutes total

### Step 2: Review Results

After fine-tuning completes, check `results/` directory:

```
results/
├── summary_metrics.csv          # Key metrics for both stages
├── summary_metrics.json         # Same metrics in JSON
├── results_summary.md           # Markdown report
│
├── stage1_doh_detection_classification_report.txt
├── stage1_doh_detection_confusion_matrix.png
├── stage1_doh_detection_predictions.csv
└── stage1_doh_detection/adapter/  # LoRA weights
    ├── adapter_config.json
    ├── adapter_model.bin
    └── ...
│
├── stage2_malicious_detection_classification_report.txt
├── stage2_malicious_detection_confusion_matrix.png
├── stage2_malicious_detection_predictions.csv
└── stage2_malicious_detection/adapter/  # LoRA weights
    ├── adapter_config.json
    ├── adapter_model.bin
    └── ...
```

### Step 3: Push to GitHub

```bash
# Option A: Using helper script
python github_push.py

# Option B: Manual git commands
git add results/
git commit -m "Add fine-tuning results - $(date)"
git push origin main
```

---

## 🚀 Efficiency Techniques Used

### 1. **8-bit Quantization (BitsAndBytes)**
- Reduces model precision to 8-bit (from full 32-bit)
- Saves up to 75% GPU memory
- Minimal accuracy loss
- Only on GPU (CUDA)

### 2. **LoRA Adaptation**
- Only 1.3% of model parameters are trainable
- Drastically reduces memory and training time
- Maintains full model capability

### 3. **Gradient Checkpointing**
- Reduces memory usage during backprop
- ~50% memory savings
- Slight (~10%) slowdown in training

### 4. **Mixed Precision (FP16)**
- Uses half-precision where possible
- GPU optimizations (TensorCores)
- Automatic precision management

### 5. **Batch Accumulation**
- Simulates larger batch sizes with limited memory
- More stable gradients
- Default: batch_size=1, grad_accum=8

---

## ⚙️ Configuration Options

Edit `src/doh_lora/config.py` before running:

```python
# Memory efficiency
BATCH_SIZE = 1              # Start with 1 on limited GPU
GRAD_ACCUM = 8             # Increase for stability

# Training
EPOCHS = 8                  # Number of epochs (reduce to 4 for quick test)
LEARNING_RATE = 2e-4       # Fine-tuning learning rate

# LoRA
LORA_R = 8                  # Rank (4-16 typical)
LORA_ALPHA = 32            # Scaling factor

# Inference  
MAX_NEW_TOKENS = 4         # Max tokens to generate per sample
```

---

## 🔍 Monitoring Training

The pipeline logs all important information:

```
2026-04-01 10:30:45 - INFO - Device: cuda
2026-04-01 10:30:46 - INFO - GPU: NVIDIA A100

Stage 1: DoH Detection
- Train: 800 samples
- Test: 200 samples
- Trainable params: 1,414,144 (0.13%)

Training...
- Epoch 1/8: Loss=2.34, LR=2.0e-04
- Epoch 2/8: Loss=1.85, LR=1.8e-04
...

Results:
- Accuracy: 0.9560
- F1 Score: 0.9540
- Latency: 12.45 ms/sample
- Throughput: 80.32 samples/sec
```

---

## 🐛 Troubleshooting

### GPU Out of Memory

```python
# In config.py
BATCH_SIZE = 1              # Don't reduce below 1
GRAD_ACCUM = 16            # Increase accumulation
LORA_R = 4                 # Reduce rank (with some accuracy loss)
MAX_LENGTH = 128           # Reduce sequence length
```

Or use CPU-only training (slower but works):

```bash
# Set before running
$env:CUDA_VISIBLE_DEVICES = "-1"
python finetune.py
```

### Missing Dependencies

```bash
pip install -r requirements.txt
pip install bitsandbytes==0.41.3  # For quantization
```

### FileNotFoundError for CSVs

Ensure files exist in `data/`:
```bash
ls data/merge_first_layer.csv
ls data/merge_second_layer.csv
```

---

## 📊 Understanding Results

### Metrics

| Metric | Meaning | Good Range |
|--------|---------|------------|
| **Accuracy** | Correct predictions / Total | 0.85+ |
| **Precision** | True positives / Predicted positives | 0.85+ |
| **Recall** | True positives / Actual positives | 0.85+ |
| **F1 Score** | Harmonic mean of precision & recall | 0.85+ |
| **Latency** | Time per sample (ms) | <20ms ideal |
| **Throughput** | Samples/second | >50 ideal |

### Example Results

```
Stage 1 (DoH Detection):
- Accuracy:   0.956
- F1 Score:   0.956
- Latency:    12.45 ms
- Throughput: 80.32 samples/sec
- Efficiency: 0.077 (F1/ms ratio)

Stage 2 (Malicious Detection):
- Accuracy:   0.943
- F1 Score:   0.925
- Latency:    11.23 ms
- Throughput: 89.05 samples/sec
- Efficiency: 0.082
```

---

## 📈 Next Steps After Pushing

1. **Create GitHub Release** (if on main branch):
   ```bash
   git tag -a v1.0.1 -m "Add fine-tuning results"
   git push origin v1.0.1
   ```

2. **Update README** with new results

3. **Create Issues** for improvements based on results

4. **Document findings** in a results markdown file

---

## 🔗 References

- **LoRA Paper**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **BitsAndBytes**: Efficient Transformers via 8-Bit Quantization
- **Gradient Checkpointing**: [HuggingFace Docs](https://huggingface.co/docs/transformers/v4.18.0/performance)

---

## 📞 Support

Having issues? Check:

1. [QUICKSTART.md](QUICKSTART.md) - 5-minute setup
2. [README.md](README.md) - Full documentation
3. GitHub Issues - Ask the community

---

**Happy fine-tuning! 🚀**
