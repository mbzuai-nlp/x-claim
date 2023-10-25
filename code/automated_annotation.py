import torch
import ipdb
import stanza
from rouge import Rouge
from evaluate import load
import numpy as np
import itertools
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_csv", type=str)
    parser.add_argument("--out_csv", type=str)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--plm", type=str, default='bert-base-multilingual-cased')
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()
    return args

def get_align_words(
        sent_src, 
        sent_tgt,
        mbert_tokenizer,
        mbert_model,
        device
    ):
    # pre-processing
    token_src = [mbert_tokenizer.tokenize(word) for word in sent_src]
    token_tgt = [mbert_tokenizer.tokenize(word) for word in sent_tgt]
    wid_src = [mbert_tokenizer.convert_tokens_to_ids(x) for x in token_src]
    wid_tgt = [mbert_tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src = mbert_tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=mbert_tokenizer.model_max_length, truncation=True)['input_ids']
    ids_tgt = mbert_tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=mbert_tokenizer.model_max_length)['input_ids']
    ids_src, ids_tgt = ids_src.to(device), ids_tgt.to(device)
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]
        
    # alignment
    align_layer = 8
    threshold = 1e-3
    mbert_model.eval()
    with torch.no_grad():
        out_src = mbert_model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = mbert_model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)
    
    align_subwords = torch.nonzero(softmax_inter, as_tuple=False).cpu().numpy()
    align_words = defaultdict(set)
    for i, j in align_subwords:
        tgt_word_idx = sub2word_map_tgt[j]
        align_words[sub2word_map_src[i]].add(tgt_word_idx)
    
    for i in align_words.keys():
        align_words[i] = list(align_words[i])
        align_words[i].sort()
    return align_words

def get_projected_span(
        align_words,
        sent_tgt, 
        src_span_s, 
        src_span_e
    ):
    tgt_span_s, tgt_span_e = 0, 0

    bed = np.zeros(len(sent_tgt))
    for idx in range(src_span_s, src_span_e+1, 1):
        if idx not in align_words:
            continue
        for tgt_idx in align_words[idx]:
            bed[tgt_idx] = 1

    # take the boundary tokens as the projected span
    tgt_span_s, tgt_span_e = None, None
    bed_list = bed.tolist()
    if 1 in bed_list:
        tgt_span_s = bed_list.index(1)
        tgt_span_e = len(bed_list) - bed_list[::-1].index(1) - 1

    if tgt_span_s is None or tgt_span_e is None:
        return (-1, -1)
    return (tgt_span_s, tgt_span_e)

if __name__ == "__main__":
    args = get_arg_parser()
    if args.gpu == -1:
        device = 'cpu'
    elif args.gpu == 42:
        device = 'cuda'
    else:
        device = 'cuda:%d'%(args.gpu)
    mbert_tokenizer = AutoTokenizer.from_pretrained(args.plm)
    mbert_model = AutoModel.from_pretrained(args.plm).to(device)
    
    bertscore = load('bertscore')
    sent_segmentation = None
    try:
        sent_segmentation = stanza.Pipeline(lang=args.lang, processors='tokenize')
    except:
        print('stanza segmentation is throwing error, probably unavailable for %s language'%(args.lang))

    df = pd.read_csv(args.inp_csv, encoding='utf-8')
    source_texts, target_texts = list(df['source'].values), list(df['target'].values)
    source_texts = [t.strip() for t in source_texts]
    target_texts = [t.strip() for t in target_texts]
    assert len(source_texts) == len(target_texts)

    count = 0
    source_tokens = []
    spans, span_start_indices, span_end_indices = [], [], []
    for (input_seq, output_seq) in tqdm(zip(source_texts, target_texts)):
        str_input_seq, str_output_seq = input_seq, output_seq
        input_seq, output_seq = input_seq.split(), output_seq.split()
        
        source_tokens.append(input_seq)

        if str_output_seq in str_input_seq:
            # output_seq in input_seq: dont use awesome align
            char_idx = str_input_seq.index(str_output_seq)
            span_s = len(str_input_seq[:char_idx].strip().split())
            span_e = span_s + len(output_seq) - 1
        else:
            # choose the sentence where the claim review needs to projected
            if sent_segmentation is not None:
                input_seq_sents = sent_segmentation(str_input_seq)
                input_seq_sents = [sent.text for sent in input_seq_sents.sentences]
            else:
                # input_seq_sents not used later on
                input_seq_sents = ['none']

            if len(input_seq_sents) > 1:
                best_sent_idx = -1
                best_score = 0

                for (sent_idx, sent) in enumerate(input_seq_sents):
                    try:
                        if args.lang:
                            bert_score = bertscore.compute(predictions=[sent], references=[str_output_seq], lang=args.lang)
                        else:
                            bert_score = bertscore.compute(predictions=[sent], references=[str_output_seq], model_type=args.plm)
                    except:
                        print('error in computing bert score')
                        continue
                    score = bert_score['recall'][0]

                    if score > best_score:
                        best_score = score
                        best_sent_idx = sent_idx
                
                if best_sent_idx == -1:
                    align_words = get_align_words(
                    sent_src = output_seq,
                    sent_tgt = input_seq,
                    mbert_tokenizer = mbert_tokenizer,
                    mbert_model = mbert_model,
                    device = device
                    )

                    span_s, span_e = get_projected_span(
                        align_words = align_words,
                        sent_tgt = input_seq,
                        src_span_s = 0,
                        src_span_e = len(output_seq)-1
                    )
                else:
                    best_sent = input_seq_sents[best_sent_idx]
                    char_idx_s = str_input_seq.index(best_sent)
                    char_idx_e = char_idx_s + len(best_sent)

                    word_idx_bs_s = len(str_input_seq[:char_idx_s].strip().split())
                    word_idx_bs_e = len(str_input_seq[:char_idx_e].strip().split())

                    input_seq_subset = input_seq[word_idx_bs_s:word_idx_bs_e]

                    align_words = get_align_words(
                        sent_src = output_seq,
                        sent_tgt = input_seq_subset,
                        mbert_tokenizer = mbert_tokenizer,
                        mbert_model = mbert_model,
                        device = device
                    )

                    span_s, span_e = get_projected_span(
                        align_words = align_words,
                        sent_tgt = input_seq_subset,
                        src_span_s = 0,
                        src_span_e = len(output_seq)-1
                    )

                    span_s = word_idx_bs_s + span_s
                    span_e = word_idx_bs_s + span_e

            elif len(input_seq_sents) == 1:
                align_words = get_align_words(
                    sent_src = output_seq,
                    sent_tgt = input_seq,
                    mbert_tokenizer = mbert_tokenizer, 
                    mbert_model = mbert_model,
                    device = device
                )

                span_s, span_e = get_projected_span(
                    align_words = align_words,
                    sent_tgt = input_seq,
                    src_span_s = 0,
                    src_span_e = len(output_seq)-1
                )
            else:
                print('found no sentence using stanza sentence segmentation pipeline.')
                ipdb.set_trace()

        if span_s == -1 or span_e == -1:
            count += 1
            continue
        
        span_start_indices.append(span_s)
        span_end_indices.append(span_e)
        spans.append(' '.join(input_seq[span_s:span_e+1]))

    print('%d span labels not projected'%(count))

    data = {
        'source': source_texts,
        'target': target_texts,
        'target_on_source': spans, 
        'source_tokens': source_tokens,
        'start_index': span_start_indices,
        'end_index': span_end_indices,
    }

    df = pd.DataFrame(data)
    df.to_csv(args.out_csv,index=None,encoding='utf-8')