TOTAL_NUM_UPDATES=25000	 	# 10 epochs through RTE for bsz 16
WARMUP_UPDATES=270	     	# 6 percent of the number of updates
LR=5e-04  	             	# Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=16        	# Batch size.

ROBERTA_PATH=$1
SAVE=$2
seed=$3
ARCH=roberta_large
pdim=$4
node=$5
prefixlen=$6
insertposition=$7
LR=$8
gq=16
mode=$9
echo $ROBERTA_PATH
echo $SAVE
echo $seed
echo $ARCH

mkdir -p ${SAVE}

CUDA_VISIBLE_DEVICES=$node fairseq-train subj-bin/ \
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
    --add-suffix --suffix-len $prefixlen --prompt-generation --generation-freeze --insert-position $insertposition --generation-layer 2 --generation-quaternions $gq --middle-prompt-insert-layer 1 --middle-previous --middle-prompt-mode layerb --phm-bottleneck-dim $pdim --prompt-insert-mode $mode \
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
