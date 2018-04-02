#!/usr/bin/env bash

DATASET=ag_news

MODELDIR=/tmp/starspace/models
DATADIR=/tmp/starspace/data

mkdir -p "${MODELDIR}"
mkdir -p "${DATADIR}"


echo "Downloading AG news dataset"
if [ ! -f "${DATADIR}/${DATASET}_csv/train.csv" ]
then
    wget -c "https://s3.amazonaws.com/fair-data/starspace/ag_news_csv.tar.gz" -O "${DATADIR}/${DATASET[0]}_csv.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET}_csv.tar.gz" -C "${DATADIR}"
  fi


echo "Start training on AG news dataset"
./starspace train \
  --train-file "${DATADIR}/${DATASET}_csv/train.csv" \
  --model-path "${MODELDIR}/${DATASET}" \
  --file-format ${DATASET} \
  --d-embed=10 \
  --lr=0.01 \
  --epochs=5


echo "Evaluate trained model"
./starspace test \
  --test-file "${DATADIR}/${DATASET}_csv/test.csv" \
  --model-path "${MODELDIR}/${DATASET}" \
  --file-format ${DATASET}
