# EmboMatrix
Code repository of the paper **EmboMatrix: A Scalable Training-Ground for Embodied Decision-Making**

# Open Source Timeline

- [ ] **Multi-Agent Driven Automated Data Factory** — Target: by 2025-10-31
- [ ] **Scalable Training Backend** — Target: by 2025-11-31
- [ ] **RL Training Architecture for EmboMatrix** — Target: by 2025-11-31

## Overview

EmboMatrix is a fully-automated data factory designed to generate task-centric scenes for embodied AI research. Built on top of the [OmniGibson](https://omnigibson.github.io/) and [Isaac Sim](https://developer.nvidia.com/isaac-sim) platforms, EmboMatrix significantly expands upon the foundations of the [Behavior-1000 dataset](https://arxiv.org/abs/2403.09227). It enables the generation of virtually unlimited volumes of high-quality, diverse scene data.

Our primary goal is to provide large-scale data for systematically investigating and validating scaling laws in embodied intelligence.

## Installation

1. **Install OmniGibson**  
   Follow the official guide: [OmniGibson Installation](https://behavior.stanford.edu/getting_started/installation.html) Download the OmniGibson dataset and Isaac Sim
2. **Set up Python Environment**  
   Create a Python 3.10 virtual environment:  
   ```bash
   conda create -n emboMatrix python=3.10
   conda activate emboMatrix
   ```

3. **Install Dependencies**  
   ```bash
   cd bddl
   pip install -e .
   
   cd ..
   pip install -r requirements.txt
   ```

## Quick Start

Follow these steps to generate task-centric scenes:

### 1. Generate Task Files
```bash
python src/bddl_gen/generate_tasks.py
```

### 2. Generate Scene Files
- **Preprocess BDDL Tasks** (organize BDDL files):  
  ```bash
  python src/preprocess_bddl_tasks.py
  ```

- **Generate Scene Files in Parallel**:  
  ```bash
  python src/generate_scenes.py
  ```

### 3. Verify Generated Data
- **Generate Verification List**:  
  ```bash
  python omnigibson/plannerdemo/verify_pipeline_compute.py
  ```

- **Run Verification Pipeline**:  
  ```bash
  bash omnigibson/plannerdemo/verify_pipeline_follow.sh
  ```

## Usage Notes

- Ensure NVIDIA Isaac Sim is properly installed and configured before running scene generation.
- The parallel scene generation step requires sufficient GPU resources.
- Generated data will be saved in the `src/data/` directory by default.

## Citation

If you use EmboMatrix in your research, please cite:  

---

*EmboMatrix is developed by [Your Team/Institution]. For issues or contributions, please visit the [GitHub repository](https://github.com/your-repo/embomatrix).*