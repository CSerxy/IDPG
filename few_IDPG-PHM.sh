ARCH=roberta_large

SAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/
NEWSAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/
ROBERTA_PATH=$SAVE'nli_large_checkpoint.pt'

K=$1
suffixlen="5"
LRs="5e-04 1e-04 5e-05 1e-05"

froms="0"
gq="16"
pdim="16"
mode="1"
seeds="1 2 3 4 5"

mkdir -p "few_results"

for LR in $LRs; do
    for from in $froms; do
        OUT_FILE='few_results/mphm-r'$K'-'$from'.txt'

        TASKs="SST-2 cr mr mpqa subj"
        insertpositions="0"
        for insertposition in $insertpositions; do
        	SUFFIX=$K'-mphmr-'$gq'-'$suffixlen'-'$insertposition'-f'$LR'_5-'
            #SAVE=$checkpoint_path'efl_roberta-d_phm-'$gq'-'$suffixlen'-'$from'/'
    
            for TASK in $TASKs; do
                for seed in $seeds; do
                    node=0
                    SAVE_FILE=$SAVE$TASK$SUFFIX$seed
                    bash scripts/few-finetune-gene_freeze-mphm.sh $ROBERTA_PATH $SAVE_FILE $seed $ARCH $node $insertposition $LR $K $TASK
                done
                wait
                for seed in $seeds; do
                    SAVE_FILE=$SAVE$TASK$SUFFIX$seed'/'
                    CUDA_VISIBLE_DEVICES=0 python 'scripts/'$TASK'_get_result.py' -i $SAVE_FILE -o $OUT_FILE -n $seed -f $from -t $insertposition -k $K -l $LR 
                done
                wait
                SAVE_FILE=$SAVE$TASK$SUFFIX
                python 'scripts/bagging_'$TASK'.py' -i $SAVE_FILE -o $OUT_FILE
                echo $TASK 'done'
            done
        done
    
        TASKs="MRPC RTE QNLI QQP"
        insertpositions="0"
        for insertposition in $insertpositions; do
            SUFFIX=$K'-mphmr-'$gq'-'$suffixlen'-'$insertposition'-f'$LR'_5-'
            for TASK in $TASKs; do
                for seed in $seeds; do
                    node=0
                    SAVE_FILE=$SAVE$TASK$SUFFIX$seed
                    bash scripts/few-finetune-gene_freeze-mphm.sh $ROBERTA_PATH $SAVE_FILE $seed $ARCH $node $insertposition $LR $K $TASK 
                done
                wait
                for seed in $seeds; do
                    SAVE_FILE=$SAVE$TASK$SUFFIX$seed'/'
                    CUDA_VISIBLE_DEVICES=0 python 'scripts/'$TASK'_get_result.py' -i $SAVE_FILE -o $OUT_FILE -n $seed -f $from -t $insertposition -k $K -l $LR 
                done
                wait
                SAVE_FILE=$SAVE$TASK$SUFFIX
                python 'scripts/bagging_'$TASK'.py' -i $SAVE_FILE -o $OUT_FILE
                echo $TASK 'done'
            done
    
            TASK="STS-B"
            for seed in $seeds; do
                node=0
                SAVE_FILE=$SAVE$TASK$SUFFIX$seed
                bash scripts/few-finetune-gene_freeze-mphm-sts-b.sh $ROBERTA_PATH $SAVE_FILE $seed $ARCH $node $insertposition $LR $K 
            done
            wait
            for seed in $seeds; do
                SAVE_FILE=$SAVE$TASK$SUFFIX$seed'/'
                CUDA_VISIBLE_DEVICES=0 python 'scripts/'$TASK'_get_result.py' -i $SAVE_FILE -o $OUT_FILE -n $seed -f $from -t $insertposition -k $K -l $LR 
            done
            wait
            SAVE_FILE=$SAVE$TASK$SUFFIX
            python 'scripts/bagging_'$TASK'.py' -i $SAVE_FILE -o $OUT_FILE
            echo $TASK 'done'
        done
    done
done
