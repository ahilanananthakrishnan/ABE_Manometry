# Whole-Colon Motility Model (Python)

This folder contains the Python implementations used to generate whole-colon stacked pressure traces for three phenotypes:

- **Healthy**: `healthy_traces.py`
- **IBS-D**: `ibsd_traces.py`
- **STC**: `stc_traces.py`

Each script simulates a 1D colon with **N = 75 segments** (proximal HAPC: 1–50, distal CMP: 51–75) and exports a tab-delimited pressure trace file compatible with plotting/analysis workflows.

---

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
