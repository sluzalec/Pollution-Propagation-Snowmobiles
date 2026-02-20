# Pollution-Propagation-Snowmobiles

This repository contains the official implementation for the paper:

**"Collocation-based Robust Variational Physics Informed Neural Network for time-dependent simulations of pollution propagation under thermal inversion conditions on Spitsbergen"** *Published/Submitted to Computers and Mathematics with Applications*

## Overview

Source code for the article "Collocation-based Robust Variational Physics Informed Neural Network for time-dependent simulations of pollution propagation under thermal inversion conditions on Spitsbergen"
This project provides the source code to reproduce the numerical experiments and visualizations described in our paper. The implementation focuses on CRVPINN running code.

## Key Features

* **Neural Network Modules**: Built using **PyTorch** (`torch.nn`).
* **Visualizations**: Static plots and dynamic animations (`Matplotlib`, `FuncAnimation`).
* **Exporting Results**: Automatic GIF/video generation for simulations (`imageio`).
* **Optimized Math**: Uses **NumPy** and **Math** for efficient calculations.

## Requirements

To run this code, you need **Python 3.8+** and the following libraries. You can install them using `pip`:

```bash
pip install torch numpy matplotlib imageio

```

*Note: The script uses standard Python libraries (`time`, `os`, `math`, `typing`, `functools`) which do not require additional installation.*

## Installation & Usage

1. **Clone the repository** (or download `source.py`):
```bash
git clone https://github.com/sluzalec/Pollution-Propagation-Snowmobiles.git
cd Pollution-Propagation-Snowmobiles

```


2. **Prepare the environment**: Ensure you have a working installation of PyTorch (CPU or GPU).
3. **Run the main script**:
```bash
python source.py

```



## Repository Structure

* `source.py` - The main executable containing the model definition, training loop, and visualization logic.
* `README.md` - Documentation of the project.
* `LICENSE` - MIT License (recommended for open-source research).

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
TBA
```

## Repository contact

**Tomasz Sluzalec** - sluzalec@agh.edu.pl

Faculty of Computer Science, AGH University

