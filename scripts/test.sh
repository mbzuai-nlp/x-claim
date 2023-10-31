#!/bin/bash
cd ../code/
gpu=0

# monolingual models
encoding=io
seed=2022
for model in mbert mdeberta xlmr; do
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
    for lang_id in hi
    do
        echo ${lang_id}
        python main.py --plm ${plm} --encoding ${encoding} --path_test ../data/test-${lang_id}.csv --weights ../ckpts/mono_${lang_id}_${model}_${encoding}_${seed}.pt --gpu ${gpu}
    done
done


# multilingual models
encoding=io
seed=2022
for model in mbert mdeberta xlmr; do
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
    for lang_id in en hi pa ta te bn
    do
        echo ${lang_id}
        python main.py --plm ${plm} --encoding ${encoding} --path_test ../data/test-${lang_id}.csv --weights ../ckpts/multilingual_${model}_${encoding}_${seed}.pt --gpu ${gpu}
    done
done


# zero-shot transfer
encoding=io
seed=2022
for model in mbert mdeberta xlmr; do
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
    for lang_id in en hi pa ta te bn
    do
        echo ${lang_id}
        python main.py --plm ${plm} --encoding ${encoding} --path_test ../data/test-${lang_id}.csv --weights ../ckpts/mono_en_${model}_${encoding}_${seed}.pt --gpu ${gpu}
    done
done


# translation models
encoding=io
seed=2022
for model in mbert mdeberta xlmr; do
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
    for lang_id in hi
    do
        echo ${lang_id}
        python main.py --plm ${plm} --encoding ${encoding} --path_test ../data/test-${lang_id}.csv --weights ../ckpts/monotrans_${lang_id}_${model}_${encoding}_${seed}.pt --gpu ${gpu}
    done
done


# encodings comparison
lang=en
model=xlmr
for seed in 2022 2023 2024; do
    for encoding in io bio beo beio
    do
        echo ${encoding}
        python main.py --plm xlm-roberta-large --encoding ${encoding} --path_test ../data/test-en.csv --weights ../ckpts/mono_${lang}_${model}_${encoding}_${seed}.pt --gpu ${gpu}
    done
done
