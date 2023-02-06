# Installation

## Installing NEMA ML4Floods on a M1 Mac

Based on notes found
[here](https://github.com/jeffheaton/t81_558_deep_learning).


### 1 Install MiniConda

Download and install miniconda from the following URL:

[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

```
# Direct package link
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg
```

Double-click to open the package and follow instructions.


### 2 Install the python environment for M1 GPU

The file [nf_mac.yml](nf_mac.yml) defines the packages needed to run
the ML4Floods and NEMA Ml4Floods systems. It is best to create an new
MiniConda Python environment:

```
# Setup a new conda environment
conda env create -f nf_mac.yml -n nf_mac

# Activate the new env
conda activate nf_mac
```

If you add a module to the YAML file, simply update the environment by
executing:

```
conda activate nf_mac
conda env update -f nf_mac.yml
```


### 3 Register the environment

This step registers the environment with Jupyter

```
python -m ipykernel install --user --name nf_mac --display-name "Python 3.9 (nf_mac)"
```


### 4 Test the environment

Run the following code in a Jupyter Notebook:

```
# What version of Python do you have?
import sys
import platform
import torch
import pandas as pd
import sklearn as sk

has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)
device = "mps" if getattr(torch,'has_mps',False) \
    else "gpu" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")
```

For a M1 Mac, you should see the message ```MPS (Apple Metal) is available```.


### 5 Install the ML4Floods package

The NEMA ML4Floods system builds on the functionality in the standard ML4Floods packaga. Install the latest version:

```
# Clone ml4floods locally
git clone https://github.com/spaceml-org/ml4floods/

# Run the installer
cd ml4floods 
python setup.py install
```
