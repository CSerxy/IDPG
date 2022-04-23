# IDPG
IDPG: An Instance-Dependent Prompt Generation Method

This repository contains the code and pre-trained models for our paper [IDPG: An Instance-Dependent Prompt Generation Method](https://arxiv.org/abs/2204.04497)

**************************** **Updates** ****************************

* 4/23: We released our [training code](#training).
* 4/9: We released [our paper](https://arxiv.org/pdf/2204.04497.pdf)
* 4/7: Our paper has been accepted to NAACL 2022 main conference!

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

