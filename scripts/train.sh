#!/bin/bash 
cd ../code/

setting=$1 #monolingual
model=$2 #xlmr
lang=$3 #en
batchsize=$4 #16

gpu=0
seed=2022
encoding=io

if [ ${model} = 'mbert' ]; then
    plm='bert-base-multilingual-cased'
elif [ ${model} = 'mdeberta' ]; then
    plm='microsoft/mdeberta-v3-base'
elif [ ${model} = 'xlmr' ]; then
    plm='xlm-roberta-large'
else
    echo "Unknown ${model} model"
    exit
fi
echo "gpu" ${gpu}
echo "seed" ${seed}
echo "encoding" ${encoding}
echo "model" ${model}
echo "pretrained encoder" ${plm}
echo "language" ${lang}
echo "training setting" ${setting}


if [ ${setting} = 'monolingual' ]; then
    # monolingual models
    python main.py --train --path_train ../data/train-${lang}.csv --path_dev ../data/dev-${lang}.csv --encoding ${encoding} --bs ${batchsize} --epochs 50 --plm ${plm} --name mono_${lang}_${model}_${encoding}_${seed} --gpu ${gpu} --seed ${seed}

elif [ ${setting} = 'multilingual' ]; then
    # multilingual models
    python main.py --train --path_train ../data/train-${lang}.csv --path_dev ../data/dev-${lang}.csv --encoding ${encoding} --bs ${batchsize} --epochs 50 --plm ${plm} --name multilingual_${model}_${encoding}_${seed} --gpu ${gpu} --seed ${seed}

elif [ ${setting} = 'translatetrain' ]; then
    # translatetrain models
    python main.py --train --path_train ../data/train-en2${lang}.csv --path_dev ../data/dev-${lang}.csv --encoding ${encoding} --bs ${batchsize} --epochs 50 --plm ${plm} --name monotrans_${lang}_${model}_${encoding}_${seed} --gpu ${gpu} --seed ${seed}
else
    echo "Unknown ${setting} training setting"
    exit
fi