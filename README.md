# geoai_train

**Status (6/20/2026):**

- Input pre-training complete
- Dashboard functional program ready

## How to use

### Step 1: Start the Server

Open a terminal with the conda environment and run:

```bash
conda activate geoai
cd d:\geoai_train\geoai_train
python Server.py
```

Then open your browser and navigate to: <http://localhost:5000>

### Step 2: Start Flood Detection (in a new terminal)

Open a **new terminal** (do not close the first one) and run:

```bash
cd d:\geoai_train\geoai_train\Flood-detection
python main.py
