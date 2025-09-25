# Political Ideology Shifts in Large Language Models

This repository contains the code used in the paper:  
**Bernardelle, P., Civelli, S., Fröhling, L., Lunardi, R., Roitero, K., & Demartini, G. (2025). Political Ideology Shifts in Large Language Models.**

## 📜 Overview

Large Language Models (LLMs) are increasingly deployed in politically sensitive settings. Our goal in this work was to understand **how persona adoption can shift the political orientation of LLM outputs**—both implicitly through thematic cues, and explicitly through direct ideological framing.

Using **200,000 synthetic personas** from the PersonaHub dataset, we evaluated seven open-source LLMs (7B–70B+ parameters) with the Political Compass Test across three studies:

1. **Implicit ideological malleability** — How baseline persona adoption shapes political expression.
2. **Explicit ideological malleability** — How direct ideological labels steer responses.
3. **Latent thematic influence** — How themes in persona descriptions (e.g., history, politics, business) induce predictable ideological deviations.

## 📌 Key Findings

- Larger models show **broader and more polarized** ideological coverage.
- Models are **more responsive to right-authoritarian** than left-libertarian priming.
- Thematic persona content acts as a **consistent directional force** in shaping political outputs.
- Ideological responsiveness is **scale-dependent**—stronger in larger models.

## 📂 Data & Results

The datasets used and produced in this work are hosted on Zenodo:  
[**Zenodo Repository — Data & Results**](https://zenodo.org/records/16869784)

Download and unzip:

- `data.zip` → Replace the `data/` folder in this repo with its contents.  
- `results.zip` → Replace the `results/` folder in this repo with its contents.

## ⚙️ Reproducing the Results

1. **Clone this repository** 
```bash
git clone https://github.com/d-lab/political-ideology-shifts-in-LLMs.git
cd political-ideology-shifts-in-LLMs
```

2. **Install dependencies**
We recommend using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Download and unzip the archives from Zenodo.**
Place and unzip them so that:
- The contents of data.zip replace the data/ folder.
- The contents of results.zip replace the results/ folder.

4 **Follow the folder structure and scripts to reproduce results**
- data/ → Contains all input files required for the experiments.
- results/ → Contains all experiment outputs.
- scripts/ → Contains scripts to process data, run experiments, and generate figures.

<!--
## 📖 Citation

If you use this code or data, please cite:

@article{bernardelle2025political,
title={Political Ideology Shifts in Large Language Models},
author={Bernardelle, Pietro and Civelli, Stefano and Fröhling, Leon and Lunardi, Riccardo and Roitero, Kevin and Demartini, Gianluca},
journal={Nature Machine Intelligence},
year={2025}
}
