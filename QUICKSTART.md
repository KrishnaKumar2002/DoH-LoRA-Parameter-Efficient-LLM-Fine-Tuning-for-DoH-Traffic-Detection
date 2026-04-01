# Quick Start Guide

Get up and running with DoH-LoRA in 5 minutes.

## Prerequisites

- **Python** 3.8 or higher
- **Git**
- 8GB+ RAM (16GB+ if using GPU)
- **Optional**: NVIDIA GPU with CUDA 11.8+

## Step-by-Step Setup

### 1. Clone & Navigate

```bash
git clone https://github.com/KrishnaKumar2002/DoH-LoRA-Parameter-Efficient-LLM-Fine-Tuning-for-DoH-Traffic-Detection.git
cd DoH-LoRA-Parameter-Efficient-LLM-Fine-Tuning-for-DoH-Traffic-Detection
```

### 2. Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected installation time**: 3-5 minutes (depending on internet speed)

### 4. Prepare Your Data

Copy your CSV files to the `data/` directory:

- `data/merge_first_layer.csv` - Network flows with DoH labels
- `data/merge_second_layer.csv` - Network flows with malicious labels

See [data/README.md](data/README.md) for format requirements.

### 5. Run the Pipeline

```bash
python -m src.doh_lora.main
```

**Expected runtime**:
- CPU: 20-30 minutes per stage
- GPU (T4): 5-10 minutes per stage
- GPU (A100): 1-2 minutes per stage

### 6. Check Results

Results are saved in `results/`:

```bash
# View summary metrics
cat results/summary_metrics.csv

# View markdown report
cat results/results_summary.md

# View stage 1 confusion matrix
results/stage1_doh_detection_confusion_matrix.png

# View stage 2 confusion matrix  
results/stage2_malicious_detection_confusion_matrix.png
```

## Common Issues

### ❌ CUDA Out of Memory

```python
# Edit src/doh_lora/config.py
Config.BATCH_SIZE = 1
Config.GRAD_ACCUM = 16
```

### ❌ ModuleNotFoundError

Ensure you're in the activated virtual environment:

```bash
# Activate it again
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### ❌ CSV Not Found

Verify files exist:

```bash
ls data/  # or dir data\ on Windows
```

Files must be named exactly:
- `merge_first_layer.csv`
- `merge_second_layer.csv`

## Next Steps

- **Customize training**: Edit `src/doh_lora/config.py`
- **Use as module**: See [README.md](README.md#-usage)
- **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Getting Help

- 📖 Full docs: [README.md](README.md)
- 🐛 Report bugs: [GitHub Issues](https://github.com/KrishnaKumar2002/DoH-LoRA-Parameter-Efficient-LLM-Fine-Tuning-for-DoH-Traffic-Detection/issues)
- 💬 Ask questions: [GitHub Discussions](https://github.com/KrishnaKumar2002/DoH-LoRA-Parameter-Efficient-LLM-Fine-Tuning-for-DoH-Traffic-Detection/discussions)

---

Done! You're ready to fine-tune. 🚀
