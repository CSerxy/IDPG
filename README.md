# IDPG
IDPG: An Instance-Dependent Prompt Generation Method

This repository contains the code for our paper [IDPG: An Instance-Dependent Prompt Generation Method](https://arxiv.org/abs/2204.04497)

**************************** **Updates** ****************************

* 4/23: We released our [training code](#training).
* 4/9: We released [our paper](https://arxiv.org/pdf/2204.04497.pdf)
* 4/7: Our paper has been accepted to NAACL 2022 main conference!

## Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install IDPG** and develop locally:

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
We also provided a GloVe-based prompt generation method, with much fewer FLOPS but maintaining similar performance. Before running this model, you need to download [GloVe embeddings](https://nlp.stanford.edu/data/glove.6B.zip) first. 

## Train and evaluate IDPG
We provide example training scripts for both settings. 

IDPG-PHM.sh, IDPG-PHM-GloVe.sh, IDPG-DNN.sh are the training scripts for full data setting. 

few_IDPG-PHM.sh, few_IDPG-PHM-GloVe.sh are the training scripts for few-shot setting. 

### Example usage
Take IDPG-PHM-GloVe.sh as an example. To run it, you need to download checkpoints and GloVe files first. Next, modify the `SAVE` and `NEWSAVE` variable, where `SAVE` is the path to your checkpoint folder, `NEWSAVE` is the path where you hope to store the intermediate checkpoints (i.e., fine-tuned checkpoints). Also, change `glove_path` to your local path as well. 

We explain the main arguments in following:
* `suffixlen`: the generated prompt length in each Transformer layer.
* `LRs`: the tuned learning rate.
* `seeds`: the multiple random seeds.
* `pdim`: the hidden layer size in phm bottleneck  

The result will be stored at the path `OUT_FILE`.

Below is an example of showing you how to train and evaluate IDPG-PHM-GloVE:
```
ARCH=roberta_large

SAVE=/PATH/TO/YOUR/CHECKPOINTS/FOLDER/
NEWSAVE=/PATH/TO/WHERE/YOU/STORE/THE/FINETUNE/CHECKPOINT/FOLDER/
glove_path=/PATH/TO/GloVe/FILES/

SAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/
NEWSAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/
glove_path=/home/zhuofeng/glove.6B.300d.txt
ROBERTA_PATH=$SAVE'roberta_large_checkpoint.pt'
#ROBERTA_PATH=$SAVE'nli_large_checkpoint.pt'

insertpositions="0"
suffixlen="5"
LRs="5e-3 1e-3 5e-4 1e-4"
seeds="1 2 3 4 5"

pdim="16"
mode="1"
mkdir -p "main_results"
OUT_FILE='main_results/IDPG-PHM-glove-layerb.txt'$pdim'-'$suffixlen


for LR in $LRs; do
    for insertposition in $insertpositions; do
        SUFFIX='-multi-phm-glove-layerb-'$mode'-'$pdim'-'$suffixlen'-'$insertposition'-f'$LR'_5-'
        TASKs='mpqa subj cr mr sst-2 qnli rte mrpc sts-b qqp'
        for TASK in $TASKs; do
            for seed in $seeds; do
                node=0
                SAVE_FILE=$NEWSAVE$TASK$SUFFIX$seed
                bash 'scripts/glove-multi-suffix-'$TASK'_finetune-phm-layerb.sh' $ROBERTA_PATH $SAVE_FILE $seed $pdim $node $suffixlen $insertposition $LR $mode $glove_path
            done
            wait
            for seed in $seeds; do
                 SAVE_FILE=$NEWSAVE$TASK$SUFFIX$seed'/'
                 CUDA_VISIBLE_DEVICES=0 python 'scripts/'$TASK'_get_result.py' -i $SAVE_FILE -o $OUT_FILE -n $seed -t $insertposition -l $LR
            done
            wait
            #SAVE_FILE=$NEWSAVE$TASK$SUFFIX
            #python 'scripts/bagging_'$TASK'.py' -i $SAVE_FILE -o $OUT_FILE
            #echo $TASK 'done'
        done
    done 
done
```
## Training Environments
The codebase is based on [fairseq](https://github.com/pytorch/fairseq). We tested our codes in an Nvidia A100 environment. We notice that the model's performance is sensitive to one's server environment and package version. You may find a slight performance difference if you do not have the exact same environment. We highly recommend you run hyper-parameter tuning in your own environment based on our sample scripts. 

## Citation
Please cite our paper if you use IDPG in your work:
```bibtex
@article{wu2022idpg,
  title={IDPG: An Instance-Dependent Prompt Generation Method},
  author={Wu, Zhuofeng and Wang, Sinong and Gu, Jiatao and Hou, Rui and Dong, Yuxiao and Vydiswaran, VG and Ma, Hao},
  journal={arXiv preprint arXiv:2204.04497},
  year={2022}
}
```
