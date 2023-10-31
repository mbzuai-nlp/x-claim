# _Lost in Translation, Found in Spans_:<br/> Identifying Claims in Multilingual Social Media

We release a dataset called **X-CLAIM** for the task of multilingual claim span identification. 
X-CLAIM consists of 7K real-world claims, and social media posts containing them, collected from various social media platforms (e.g., Instagram) in English, Hindi, Punjabi, Tamil, Telugu and Bengali.
We also provide the code of baseline model that trains the encoder-only language models like mDeBERTa on the X-CLAIM dataset.

This work got accepted at the **EMNLP 2023 (main)** conference.

<a href='https://arxiv.org/abs/2310.18205'><img src='https://img.shields.io/badge/ArXiv-PDF-blue'></a>

Authors: [Shubham Mittal](https://scholar.google.com/citations?view_op=list_works&hl=en&authuser=1&hl=en&user=l_bIdRcAAAAJ&authuser=1), [Megha Sundriyal](https://scholar.google.com/citations?hl=en&authuser=1&user=vbmdVSAAAAAJ), [Preslav Nakov](https://scholar.google.com/citations?hl=en&authuser=1&user=DfXsKZ4AAAAJ).


## X-CLAIM Dataset
The `train`, `dev` and `test` split for the `lang` language is provided inside `./data/` folder in the file named `./data/{split}-{lang}.csv`. Each file contains three columns:
1. `tokens`: list of tokens in the social media post's text
2. `span_start_index`: starting token index of claim span in tokens list
3. `span_end_index`: ending token index (included) of claim span in tokens list

For reproducibility of the results, we provide the translated data in the target `lang` language in `./data/{split}-en2{lang}.csv` files for the `train` and `dev` splits. Note that the `dev` split is only provided for Telugu and Bengali. 

We use language IDs in the `lang` variable as per below scheme.
1. English: en
2. Hindi: hi
3. Punjabi: pa
4. Tamil: ta
5. Telugu: te
6. Bengali: bn

To reproduce the multilingual training baselines, the `./data/{split}-multilingual.csv` file contains the aggregated data of all languages for the `train` and `dev` splits.


## Installation

Run the below commands to install the required dependencies. 

```
conda create --name xclaim python=3.7
conda activate xclaim
pip install -r requirements.txt
```



## Data Curation
Our two-step methodology to create the X-CLAIM dataset.

![pipeline](https://github.com/mbzuai-nlp/x-claim/assets/65343158/f482b292-aa61-4dfb-915b-84439c577fc0)


Here, we provide the command to run the **automated annotation** step for marking the claim span in the social media post text using the normalized claim.


```
cd ./code/
python automated_annotation.py --inp_csv <input_csv_file_path> --out_csv <output_csv_file_path>  --gpu <gpu id>
```

`<input_csv_file_path>` file should contain two columns:
1. `source`: the text (e.g., social media post text) onto which the target text is mapped
2. `target`: the text (e.g., normalized claim) which is mapped onto the source text

`<output_csv_file_path>` file will contain four columns (in addition to `source` and `target` columns):
1. `target_on_source`: the target text mapped onto the source text, i.e., the claim span in social media post created from the normalized claim
2. `source_tokens`: list of the tokens in source text
3. `start_index`: starting token index of the `target_on_source` text
4. `end_index`: ending token index (included) of the `target_on_source` text


## Model
Train encoder-only language models such as mBERT, mDeBERTa and XLM-R on the X-CLAIM dataset using the below instructions.

### Training
Run the below command with following variables: `setting`, `model`, `lang` and `batchsize`.

```
cd ./scripts/
sh train.sh <setting> <model> <lang> <batchsize>
```

1. `setting` = monolingual, multilingual or translatetrain
2. `model` = mbert, mdeberta or xlmr
3. `lang` = en, hi, pa, ta, te, bn or multilingual
4. `batchsize` = 16 (for xlmr) or 32 (for mbert and mdeberta)


Example command for training our best baseline model, Multilingual mDeBERTa on the data containing training data of all the languages 

```
cd ./scripts/
sh train.sh multilingual mdeberta multilingual 32
```

All experiments are run with a single A100 (40GB) GPU.

### Evaluating
Use the below command to evaluate a model, say the above trained Multilingual mDeBERTa checkpoint, on the `lang` language.

```
cd ./code/
python main.py --plm microsoft/mdeberta-v3-base --path_test ../data/test-<lang>.csv --weights <path to model checkpoint>
```

The commands to evaluate the models in different settings like zero-shot transfer or on multiple languages are provided in `./scripts/test.sh`.

## Cite
Please cite our work if you use or extend our work:
```
@misc{mittal2023xclaim,
      title={{L}ost in {T}ranslation, {F}ound in {S}pans: {I}dentifying {C}laims in {M}ultilingual {S}ocial {M}edia}, 
      author={Shubham Mittal and Megha Sundriyal and Preslav Nakov},
      year={2023},
      eprint={2310.18205},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
