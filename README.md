# BoDA Runner

This repository contains a lightweight script to run the MDLT training code using the BoDA algorithm.

## Installation

1. Create and activate a Python environment (optional).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running

Execute the helper script from the repository root:

```bash
bash run_boda.sh [DATASET] [OUTPUT_NAME] [DATA_DIR] [OUTPUT_DIR]
```

Arguments are optional:

- `DATASET` - Dataset name (default: `PACS`)
- `OUTPUT_NAME` - Folder name for the experiment (default: `boda_experiment`)
- `DATA_DIR` - Path to the dataset root (default: `./data`)
- `OUTPUT_DIR` - Directory to store outputs (default: `./output`)

The script launches `python -m mdlt.train` with the BoDA algorithm and logs results in the specified output directory.

