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
conda create -n IDPG python=3.6
conda activate IDPG
git clone https://github.com/CSerxy/IDPG.git
cd IDPG 
pip install --editable ./
pip install requests
pip install pytorch-metric-learning==0.9.90.dev0

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# on A100:
# conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
# pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pre-trained models
We mainly trained our model starting from below two checkpoints. We used roberta.large for main results, roberta.large.mnli for few-shot results. 

Model | Description | # params | Download
---|---|---|---
`roberta.large` | RoBERTa using the BERT-large architecture | 355M | [roberta.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz)
`roberta.large.mnli` | `roberta.large` finetuned on [MNLI](http://www.nyu.edu/projects/bowman/multinli) | 355M | [roberta.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz)

### GloVe 
We also provided a GloVe-based prompt generation method, with much less FLOPS but maintain a similar performance. Before running this model, you need to download [GloVe embeddings](https://nlp.stanford.edu/data/glove.6B.zip) first. 
