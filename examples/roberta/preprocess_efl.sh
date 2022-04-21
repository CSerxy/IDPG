#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# raw glue data as downloaded by glue download script (https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
if [[ $# -ne 2 ]]; then
  echo "Run as following:"
  echo "./examples/roberta/preprocess_GLUE_tasks.sh <glud_data_folder> <task_name> <K>"
  exit 1
fi

GLUE_DATA_FOLDER=$1

# download bpe encoder.json, vocabulary and fairseq dictionary
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

TASKS=$2 # QQP

K=$3

if [ "$TASKS" = "ALL" ]
then
  TASKS="cr mr subj mpqa trec agnews"
fi

for TASK in $TASKS
do
  echo "Preprocessing $TASK"

  TASK_DATA_FOLDER="$GLUE_DATA_FOLDER/$TASK"
  echo "Raw data as downloaded from glue website: $TASK_DATA_FOLDER"

  SPLITS="train dev test"
  INPUT_COUNT=1

  # Strip out header and filter lines that don't have expected number of fields.
  rm -rf "$TASK_DATA_FOLDER/processed"
  mkdir -p "$TASK_DATA_FOLDER/processed"
  for SPLIT in $SPLITS
  do
    cp "$TASK_DATA_FOLDER/$SPLIT$K.csv" "$TASK_DATA_FOLDER/processed/$SPLIT$K.csv";
  done

  # Split into input0, input1 and label
  for SPLIT in $SPLITS
  do
    cut -c3- "$TASK_DATA_FOLDER/processed/$SPLIT$K.csv" > "$TASK_DATA_FOLDER/processed/$SPLIT$K.raw.input0";
    cut -c1 "$TASK_DATA_FOLDER/processed/$SPLIT$K.csv" > "$TASK_DATA_FOLDER/processed/$SPLIT$K.label";

    # BPE encode.
    for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
    do
      LANG="input$INPUT_TYPE"
      echo "BPE encoding $SPLIT/$LANG"
      python -m examples.roberta.multiprocessing_bpe_encoder \
      --encoder-json encoder.json \
      --vocab-bpe vocab.bpe \
      --inputs "$TASK_DATA_FOLDER/processed/$SPLIT$K.raw.$LANG" \
      --outputs "$TASK_DATA_FOLDER/processed/$SPLIT$K.$LANG" \
      --workers 60 \
      --keep-empty;
    done
  done

  # Remove output directory.
  rm -rf "$TASK-bin$K"

  DEVPREF="$TASK_DATA_FOLDER/processed/dev$K.LANG"
  TESTPREF="$TASK_DATA_FOLDER/processed/test$K.LANG"

  # Run fairseq preprocessing:
  for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
  do
    LANG="input$INPUT_TYPE"
    fairseq-preprocess \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train$K.$LANG" \
      --validpref "${DEVPREF//LANG/$LANG}" \
      --testpref "${TESTPREF//LANG/$LANG}" \
      --destdir "$TASK-bin$K/$LANG" \
      --workers 60 \
      --srcdict dict.txt;
  done
  if [[ "$TASK" !=  "STS-B" ]]
  then
    fairseq-preprocess \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train$K.label" \
      --validpref "${DEVPREF//LANG/label}" \
      --testpref "${TESTPREF//LANG/label}" \
      --destdir "$TASK-bin$K/label" \
      --workers 60;
  else
    # For STS-B output range is converted to be between: [0.0, 1.0]
    mkdir -p "$TASK-bin/label"
    awk '{print $1 / 5.0 }' "$TASK_DATA_FOLDER/processed/train$K.label" > "$TASK-bin$K/label/train$K.label"
    awk '{print $1 / 5.0 }' "$TASK_DATA_FOLDER/processed/dev$K.label" > "$TASK-bin$K/label/valid$K.label"
  fi
done
