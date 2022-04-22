ARCH=roberta_large

SAVE=/PATH/TO/YOUR/CHECKPOINTS/FOLDER/
NEWSAVE=/PATH/TO/WHERE/YOU/STORE/THE/FINETUNE/CHECKPOINT/FOLDER/

SAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/
NEWSAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/
ROBERTA_PATH=$SAVE'roberta_large_checkpoint.pt'
#ROBERTA_PATH=$SAVE'nli_large_checkpoint.pt'

insertpositions="0"
suffixlen="5"
LRs="5e-3 1e-3 5e-4 1e-4"
#LRs="5e-4"
seeds="1 2 3 4 5"
#seeds="1"

pdim="16"
mode="1"
mkdir -p "main_results"
OUT_FILE='main_results/IDPG-DNN-p-layerb.txt'$pdim'-'$suffixlen


for LR in $LRs; do
    for insertposition in $insertpositions; do
        SUFFIX='-multi-dnn-p-layerb-'$mode'-'$pdim'-'$suffixlen'-'$insertposition'-f'$LR'_5-'
        TASKs='mpqa subj cr mr sst-2 qnli rte mrpc sts-b qqp'
        for TASK in $TASKs; do
            for seed in $seeds; do
                node=0
                SAVE_FILE=$NEWSAVE$TASK$SUFFIX$seed
                bash 'scripts/multi-suffix-'$TASK'_finetune-dnn-p-layerb.sh' $ROBERTA_PATH $SAVE_FILE $seed $pdim $node $suffixlen $insertposition $LR $mode 
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
