# Data Directory

This directory should contain your CSV files for the fine-tuning pipeline.

## Required Files

Place the following CSV files in this directory:

### 1. `merge_first_layer.csv`
**Purpose**: Stage 1 - DoH Detection

**Required Columns**:
- **Target**: `DoH` (boolean column - True/False or 1/0)
- **Features**: Numeric columns representing network flow characteristics
  - Examples: packet counts, byte counts, durations, protocols, etc.
- **Exclude**: Columns named `SourceIP`, `DestinationIP`, `TimeStamp`

**Example**:
```csv
SourceIP,DestinationIP,Protocol,PacketCount,ByteCount,Duration,DoH
192.168.1.1,8.8.8.8,UDP,24,546,0.234,True
10.0.0.5,1.1.1.1,TCP,12,1024,0.458,False
```

### 2. `merge_second_layer.csv`
**Purpose**: Stage 2 - Malicious Traffic Detection

**Required Columns**:
- **Target**: `Label` (string column - must be lowercase: "malicious" or "benign")
- **Features**: Same network flow characteristics
- **Exclude**: Same IP and timestamp columns

**Example**:
```csv
SourceIP,DestinationIP,Protocol,PacketCount,ByteCount,Duration,Label
192.168.1.100,45.33.32.156,TCP,156,8956,12.3,malicious
10.0.0.50,13.107.42.14,TCP,45,3245,6.7,benign
```

## Usage

1. Prepare your data and ensure it matches the expected format
2. Place files in this directory
3. Run the pipeline:
   ```bash
   python -m src.doh_lora.main
   ```

## Data Guidelines

- **Size**: Minimum 100 samples per class (recommend 1000+)
- **Balance**: Try to balance classes (50/50 if possible, but model handles imbalance)
- **Features**: Include numeric predictive features, remove identifiers
- **Missing Values**: Clean data before including (remove rows with NaN)

## Privacy & Security

⚠️ **Important**: Do not commit actual CSV files to git!

- Your `.gitignore` includes `data/*.csv` to prevent accidental commits
- These files may contain sensitive network data
- Always review before sharing

---

For questions about data format, see [README.md](../README.md#-usage) or open an issue.
