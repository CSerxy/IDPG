ARCH=roberta_large

SAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/
NEWSAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/
ROBERTA_PATH=$SAVE'roberta_large_checkpoint.pt'
#ROBERTA_PATH=$SAVE'nli_large_checkpoint.pt'
#suffixlens="5"
#insertpositions=$1
#simply=$2
#LR="5e-4"

insertpositions="0"
suffixlen="5"
LRs="5e-3 1e-3 5e-4 1e-4"
LRs="5e-4"
seeds="1 2 3 4 5"
seeds="1"

pdim="16"
mode="1"
mkdir -p "main_results"
OUT_FILE='main_results/sanity-p-ss-layerb'$mode'.txt'$pdim'-'$suffixlen

for LR in $LRs; do
    for insertposition in $insertpositions; do
        SUFFIX='-sanity-phm-p-layerb-'$mode'-'$pdim'-'$suffixlen'-'$insertposition'-f'$LR'_5-'
        TASKs='mpqa subj cr mr sst-2 qnli rte mrpc sts-b qqp'
        TASKs='rte'
        for TASK in $TASKs; do
            for seed in $seeds; do
                node=0
                SAVE_FILE=$NEWSAVE$TASK$SUFFIX$seed
                bash 'scripts/multi-suffix-'$TASK'_finetune-phm-p-layerb.sh' $ROBERTA_PATH $SAVE_FILE $seed $pdim $node $suffixlen $insertposition $LR $mode
            done
            wait
        done
    done 
done
