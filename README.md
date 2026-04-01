# DoH-LoRA: Parameter-Efficient Fine-Tuning for DoH Traffic Detection

A production-ready Python framework for fine-tuning small language models using **LoRA** (Low-Rank Adaptation) to detect DNS-over-HTTPS (DoH) traffic and classify malicious network flows. Designed for efficient inference on resource-constrained devices.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Output & Results](#-output--results)
- [Development](#-development)
- [License](#-license)

---

## ✨ Features

- **Two-Stage Pipeline**: 
  - Stage 1: Binary classification of DoH vs. non-DoH traffic
  - Stage 2: Malicious vs. benign traffic classification

- **Parameter-Efficient**: Uses LoRA adapters (~1.3% trainable parameters)
- **Production Ready**: Modular, well-tested, documented code
- **Metric-Rich**: Accuracy, precision, recall, F1, latency, throughput
- **GPU Optimized**: Mixed precision training, gradient checkpointing
- **Easy Reproducibility**: Fixed seeds, configuration management

---

## 🏗️ Architecture

### Two-Layer Classification Pipeline

```
Network Traffic Data
        ↓
    [Stage 1: DoH Detection]
    ├─ Input: Network flow features
    ├─ Model: TinyLlama-1.1B with LoRA
    ├─ Output: "doh" or "not_doh"
    └─ Result → Send to Stage 2 if DoH detected
        ↓
    [Stage 2: Malicious Detection]
    ├─ Input: Network flow features (if DoH)
    ├─ Model: TinyLlama-1.1B with LoRA
    └─ Output: "malicious" or "benign"
```

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| r (rank)  | 8     |
| alpha     | 32    |
| dropout   | 0.05  |
| target modules | `q_proj`, `v_proj` |
| trainable % | ~1.3% |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU) or CPU
- 8GB+ RAM (16GB+ recommended for fine-tuning)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KrishnaKumar2002/DoH-LoRA-Parameter-Efficient-LLM-Fine-Tuning-for-DoH-Traffic-Detection.git
   cd DoH-LoRA-Parameter-Efficient-LLM-Fine-Tuning-for-DoH-Traffic-Detection
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**:
   - Place `merge_first_layer.csv` in `data/`
   - Place `merge_second_layer.csv` in `data/`
   
   Expected columns:
   - **First layer**: Network metrics + `DoH` column (boolean)
   - **Second layer**: Network metrics + `Label` column (string: "malicious" / "benign")

5. **Run the pipeline**:
   ```bash
   python -m src.doh_lora.main
   ```

---

## 📁 Project Structure

```
DoH-LoRA/
├── src/
│   └── doh_lora/
│       ├── __init__.py           # Package initialization
│       ├── main.py               # Entry point
│       ├── pipeline.py           # Main orchestration pipeline
│       ├── config.py             # Configuration management
│       ├── data.py               # Data loading & dataset classes
│       ├── model.py              # Model building & training
│       ├── evaluation.py         # Metrics & evaluation
│       └── utils.py              # Utility functions
├── data/
│   ├── merge_first_layer.csv     # [NOT IN REPO] Place your data here
│   └── merge_second_layer.csv    # [NOT IN REPO] Place your data here
├── results/
│   ├── summary_metrics.csv       # Aggregated metrics
│   ├── summary_metrics.json      # JSON format metrics
│   ├── results_summary.md        # Summary report
│   ├── stage1_doh_detection_*.* # Stage 1 outputs
│   └── stage2_malicious_detection_*.* # Stage 2 outputs
├── tests/                        # Unit tests
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── README.md                     # This file
└── LICENSE                       # MIT License
```

---

## ⚙️ Configuration

Edit `src/doh_lora/config.py` to customize:

### Model & Training

```python
# Model
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# LoRA
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
BATCH_SIZE = 1
EPOCHS = 8
LEARNING_RATE = 2e-4
GRAD_ACCUM = 8
```

### Inference

```python
MAX_NEW_TOKENS = 4
BATCH_SIZE_EVAL = 8
```

### Paths (Environment-based)

```bash
export FIRST_LAYER_CSV="path/to/merge_first_layer.csv"
export SECOND_LAYER_CSV="path/to/merge_second_layer.csv"
export LOG_LEVEL="INFO"
```

---

## 📊 Usage

### Run Full Pipeline

```bash
python -m src.doh_lora.main
```

### Using as a Module

```python
from src.doh_lora.pipeline import run_pipeline
from src.doh_lora.config import Config

# Customize config
Config.EPOCHS = 4
Config.BATCH_SIZE = 2

# Run
run_pipeline()
```

### Programmatic Training

```python
from src.doh_lora.pipeline import train_and_evaluate_task
from src.doh_lora.config import Config
import pandas as pd

# Load data
df = pd.read_csv("data/merge_first_layer.csv")

# Train
metrics = train_and_evaluate_task(
    df=df,
    target_col="DoH",
    feature_cols=["feature1", "feature2", ...],
    task_name="custom_task",
    classes=["doh", "not_doh"],
    output_dir="results/custom/",
    positive_label="doh",
)

print(metrics["test_f1"])
```

---

## 📈 Output & Results

### File Structure

```
results/
├── summary_metrics.csv
│   └── Aggregated performance metrics for both stages
├── summary_metrics.json
│   └── Same metrics in JSON format
├── results_summary.md
│   └── Markdown report with findings
│
├── stage1_doh_detection_classification_report.txt
├── stage1_doh_detection_confusion_matrix.png
├── stage1_doh_detection_predictions.csv
└── stage1_doh_detection/adapter/
    ├── adapter_config.json
    ├── adapter_model.bin
    └── tokenizer files...
│
├── stage2_malicious_detection_classification_report.txt
├── stage2_malicious_detection_confusion_matrix.png
├── stage2_malicious_detection_predictions.csv
└── stage2_malicious_detection/adapter/
    ├── adapter_config.json
    ├── adapter_model.bin
    └── tokenizer files...
```

### Key Metrics

- **Accuracy**: Overall classification correctness
- **Precision**: True positives / Total predicted positives
- **Recall**: True positives / Total actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Latency**: Average time per sample inference (ms)
- **Throughput**: Samples per second
- **Efficiency**: F1 / latency ratio

### Example Output

```
task                          stage1_doh_detection
train_rows                              1000
test_rows                                250
trainable_params                      1414144
total_params                        1070000000
trainable_pct                            0.13
train_time_sec                          45.23
train_samples_per_sec                  22.10
train_peak_gpu_gb                      8.234
test_accuracy                           0.956
test_precision                          0.951
test_recall                             0.961
test_f1                                 0.956
eval_latency_ms_per_sample              12.45
eval_samples_per_sec                   80.32
efficiency_score_f1_per_ms              0.077
```

---

## 🔧 Development

### Run Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint
flake8 src/

# Type check
mypy src/
```

### Adding New Features

1. Create new module in `src/doh_lora/`
2. Add tests in `tests/`
3. Update documentation
4. Ensure all tests pass

---

## 📦 Dependencies

### Core

- **torch** (2.1.0): Deep learning framework
- **transformers** (4.44.0): HuggingFace models
- **peft** (0.12.0): Parameter-efficient fine-tuning
- **accelerate** (0.34.2): Distributed training

### Data & ML

- **pandas** (2.2.0): Data manipulation
- **scikit-learn** (1.5.1): ML metrics
- **numpy** (1.24.3): Numerical computing

### Visualization

- **matplotlib** (3.9.0): Plotting
- **seaborn** (0.13.2): Statistical plots

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Krishna Kumar M**

- GitHub: [@KrishnaKumar2002](https://github.com/KrishnaKumar2002)

---

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- [HuggingFace PEFT](https://github.com/huggingface/peft)

---

## ⚡ Performance Tips

- **GPU Memory**: Use gradient checkpointing (enabled by default)
- **Mixed Precision**: FP16 on CUDA (enabled by default)
- **Batch Size**: Increase for throughput, decrease to fit memory
- **Sequences**: Longer sequences = higher memory usage

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
Config.BATCH_SIZE = 1
Config.GRAD_ACCUM = 16  # Increase accumulation steps
```

### Slow Training

- Check GPU utilization: `nvidia-smi`
- Increase batch size if GPU memory allows
- Reduce sequence length via `Config.MAX_LENGTH`

### Missing Data

Ensure CSV files have columns:
- **Stage 1**: Target column "DoH" (boolean)
- **Stage 2**: Target column "Label" (string)

---

## 📞 Support

For issues, questions, or suggestions:
- Open an [Issue](https://github.com/KrishnaKumar2002/DoH-LoRA-Parameter-Efficient-LLM-Fine-Tuning-for-DoH-Traffic-Detection/issues)
- Create a [Discussion](https://github.com/KrishnaKumar2002/DoH-LoRA-Parameter-Efficient-LLM-Fine-Tuning-for-DoH-Traffic-Detection/discussions)

---

**Made with ❤️ for efficient network traffic classification**
