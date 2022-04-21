WARMUP_UPDATES=180	     	# 6 percent of the number of updates
NUM_CLASSES=2
MAX_SENTENCES=16        	# Batch size.

ROBERTA_PATH=$1
SAVE=$2
seed=$3
ARCH=$4
node=$5
insertposition=$6
LR=$7
K=$8
TASK=$9

gq=16
prefixlen=5

if [ $K = "100" ]
then
    TOTAL_NUM_UPDATES=313
elif [ $K = "500" ]
then
    TOTAL_NUM_UPDATES=1563
else
    TOTAL_NUM_UPDATES=3125
fi

echo $TOTAL_NUM_UPDATES

echo $TASK-bin$K

mkdir -p ${SAVE}

#--lr-scheduler fixed --lr $LR --max-update $TOTAL_NUM_UPDATES \
#--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
CUDA_VISIBLE_DEVICES=$node fairseq-train $TASK-bin$K/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch $ARCH \
    --criterion sentence_prediction \
    --freeze-encoder \
    --add-suffix --suffix-len $prefixlen --prompt-generation --generation-freeze --insert-position $insertposition --generation-layer 2 --generation-quaternions $gq --middle-prompt-insert-layer 1 --middle-prompt-mode layerb \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --lr-scheduler fixed --lr $LR --max-update $TOTAL_NUM_UPDATES \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --seed ${seed} \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 50 \
    --find-unused-parameters \
    --save-dir ${SAVE} \
    --no-epoch-checkpoints --no-last-checkpoints \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;
