# 🚀 DoH-LoRA Complete Execution Workflow

## One-Command Execution

Run everything with a single command:

```bash
python execute_workflow.py
```

This will:
1. ✅ Verify all data files and setup
2. ✅ Install all dependencies (including optional 8-bit quantization)
3. ✅ Run the two-stage fine-tuning pipeline
4. ✅ Generate comprehensive report
5. ✅ Ask if you want to push results to GitHub
6. ✅ Optionally commit and push automatically

---

## 📊 What Gets Generated

### Results Directory (`results/`)

```
results/
├── summary_metrics.csv           # Excel-friendly metrics
├── summary_metrics.json          # JSON format metrics
├── results_summary.md            # Markdown report
│
├── stage1_doh_detection/
│   ├── adapter/                  # LoRA weights (reusable)
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin
│   │   └── tokenizer_*
│   ├── checkpoints/              # Training checkpoints
│   ├── classification_report.txt # Detailed metrics
│   ├── confusion_matrix.png      # Visual confusion matrix
│   └── predictions.csv           # Model predictions on test set
│
└── stage2_malicious_detection/   # Same structure as stage1
    ├── adapter/
    ├── checkpoints/
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── predictions.csv
```

### Key Files

| File | Purpose |
|------|---------|
| `summary_metrics.csv` | Aggregated performance metrics - easy to import to spreadsheet |
| `*_confusion_matrix.png` | Visual representation of classification performance |
| `*_classification_report.txt` | Precision, recall, F1 per class |
| `*_predictions.csv` | Individual predictions - great for error analysis |
| `adapter/` | Saved LoRA weights - can be reused for inference |

---

## 🎯 Efficiency Improvements

### Memory Savings

| Technique | Memory Saving | Impact |
|-----------|---------------|--------|
| 8-bit Quantization | ~75% | Huge reduction, minimal accuracy loss |
| LoRA | ~98% (only 1.3% trainable) | Massive param reduction |
| Gradient Checkpointing | ~50% (during training) | Trades memory for compute |
| Mixed Precision (FP16) | ~50% (inference) | Auto-optimized by GPU |

### Combined Impact
- **Without optimizations**: ~16 GB GPU memory needed
- **With all optimizations**: ~2-4 GB GPU memory needed
- **Training time**: 30-45 minutes on T4 GPU

### Code References

All optimizations are in `src/doh_lora/`:
- `model.py`: 8-bit quantization with BitsAndBytes
- `config.py`: Gradient accumulation & checkpointing
- `pipeline.py`: Mixed precision (FP16) training

---

## 📈 Monitoring Training

The script logs everything to console. Look for:

```
2026-04-01 10:30:00 - INFO - ============================================================
2026-04-01 10:30:00 - INFO - DoH-LoRA Fine-Tuning Pipeline
2026-04-01 10:30:00 - INFO - ============================================================
2026-04-01 10:30:01 - INFO - Device: cuda
2026-04-01 10:30:01 - INFO - GPU: NVIDIA A100 (80GB)
2026-04-01 10:30:05 - INFO - Stage 1 shape: (1200, 45)
2026-04-01 10:30:05 - INFO - Stage 1 labels: {'doh': 700, 'not_doh': 500}
2026-04-01 10:31:12 - INFO - Task stage1_doh_detection - F1: 0.9540
2026-04-01 10:32:45 - INFO - Task stage2_malicious_detection - F1: 0.9250
```

---

## 🔄 GitHub Integration

### Automatic (Built-in)

```bash
python execute_workflow.py
# At the end:
# Do you want to push results to GitHub now? (yes/no): yes
```

### Manual (If you prefer)

```bash
# Add results
git add results/

# Commit with timestamp
git commit -m "Add fine-tuning results - $(date)"

# Push to GitHub
git push origin main
```

### Helper Script

```bash
python github_push.py
# Same as manual but with better prompts and checks
```

---

## 📊 Understanding Your Results

### Confusion Matrix

Each stage generates a confusion matrix PNG showing:
```
                 Predicted
              doh    not_doh
Actual   doh  ✓✓✓      ✗
       not_doh ✗       ✓✓✓
```

Perfect diagonal = Perfect classifier
Off-diagonal = Misclassifications

### Classification Report (TXT File)

```
              precision  recall  f1-score  support

        doh       0.956    0.961    0.958      200
    not_doh       0.951    0.945    0.948      150

   accuracy                         0.954      350
  macro avg       0.953    0.953    0.953      350
weighted avg      0.954    0.954    0.954      350
```

### CSV Metrics

Check `summary_metrics.csv` - each row is one stage:

| Column | Meaning |
|--------|---------|
| `train_time_sec` | How long training took |
| `test_f1` | F1 score on test set |
| `eval_latency_ms_per_sample` | Average time per prediction |
| `eval_samples_per_sec` | Throughput for inference |
| `efficiency_score_f1_per_ms` | Efficiency metric (higher = better) |

---

## ⚙️ Customization

Edit `src/doh_lora/config.py` before running:

```python
# Quick test (change these for faster testing)
EPOCHS = 2              # Was 8 - faster for testing
BATCH_SIZE = 2          # Was 1 - if you have GPU memory
TEST_SIZE = 0.5         # Was 0.2 - use more data for testing

# Memory constraints
BATCH_SIZE = 1          # Start here if OOM
GRAD_ACCUM = 16         # Increase this instead
LORA_R = 4              # Smaller rank = less memory (some accuracy loss)

# Inference optimization
MAX_NEW_TOKENS = 2      # Smaller = faster inference (if output is short)
```

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

```bash
pip install pandas numpy torch transformers peft accelerate
```

### "CUDA out of memory"

Option 1: Reduce batch size parameters in config.py
Option 2: Use CPU (will be slow):
```bash
set CUDA_VISIBLE_DEVICES=-1
python execute_workflow.py
```

### "FileNotFoundError: merge_first_layer.csv"

Ensure CSV files are in `data/` directory:
```bash
ls data/merge_first_layer.csv
ls data/merge_second_layer.csv
```

### No results generated

Check console for errors. Common issues:
1. Insufficient disk space
2. CSV format incompatible (missing "DoH" or "Label" columns)
3. All samples filtered out (no valid labels)

---

## 📋 Complete Workflow

```bash
# 1. Single command does everything:
python execute_workflow.py

# 2. It will:
#    - Check data files exist
#    - Install dependencies
#    - Run fine-tuning (10-45 mins)
#    - Generate results
#    - Ask about GitHub push

# 3. If you said "yes" to push:
#    - Results are committed
#    - Results are pushed to GitHub

# 4. Check results:
open results/summary_metrics.csv        # On Windows
xdg-open results/summary_metrics.csv    # On Linux
open results/                           # On Mac - opens folder
```

---

## 🎓 Learning Resources

- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Quantization**: https://arxiv.org/abs/2305.17333
- **BitsAndBytes**: https://github.com/TimDettmers/bitsandbytes
- **HuggingFace Fine-tuning**: https://huggingface.co/docs/transformers/training

---

## 📞 Need Help?

1. Check [QUICKSTART.md](QUICKSTART.md) - 5 minute setup
2. Check [README.md](README.md) - Full documentation
3. Check [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - Training details
4. Check [CONTRIBUTING.md](CONTRIBUTING.md) - Development setup

---

## ✅ Verification Checklist

Before running `python execute_workflow.py`:

- [ ] CSV files in `data/` directory
- [ ] Python 3.8+ installed
- [ ] Internet connection for downloading model
- [ ] 4-8GB free disk space for model + results
- [ ] For GPU: 4GB+ VRAM (8GB recommended)

---

## 🚀 Ready to Start?

```bash
python execute_workflow.py
```

That's it! Everything else is automated.

---

**Happy fine-tuning!** 🎉
