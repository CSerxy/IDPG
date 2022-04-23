# IDPG
IDPG: An Instance-Dependent Prompt Generation Method

This repository contains the code and pre-trained models for our paper IDPG: An Instance-Dependent Prompt Generation Method

## Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

```bash
conda create -n myenv python=3.6
conda activate myenv
git clone https://github.com/CSerxy/IDPG.git
cd IDPG 
pip install --editable ./
pip install requests
pip install pytorch-metric-learning==0.9.90.dev0

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

