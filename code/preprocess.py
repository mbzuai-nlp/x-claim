import argparse
import os
import json
import logging
import numpy as np
import pandas as pd
import ast
import ipdb
from tqdm import tqdm

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_csv", type=str, default='../data/train-en.csv')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arg_parser()

    dict_count = {
        'long_len_text': 0,
        'short_len_claim': 0,
        'no_claim_text': 0,
    }
    
    data = pd.read_csv(args.inp_csv, encoding='utf-8')
    input_seqs, span_starts, span_ends = [], [], []
    for i in range(len(data)):
        seq = ast.literal_eval((data['tokens'][i]))

        if len(seq) >=256:
            dict_count['long_len_text'] += 1
            continue

        span_s = json.loads(data['span_start_index'][i])
        span_e = json.loads(data['span_end_index'][i])
        assert len(span_s) == len(span_e), ipdb.set_trace()

        idx_span_s, idx_span_e = [], []
        for (s, e) in zip(span_s, span_e):
            if e-s+1 < 3:
                dict_count['short_len_claim'] += 1
                continue
            idx_span_s.append(s)
            idx_span_e.append(e)
        assert len(idx_span_s) == len(idx_span_e)

        if len(idx_span_e) == 0:
            dict_count['no_claim_text'] += 1
            continue

        input_seqs.append(seq)
        span_starts.append(idx_span_s)
        span_ends.append(idx_span_e)

    print(args.inp_csv)
    for (k,v) in dict_count.items():
        print('%s removed: %d/%d'%(k, v, len(data)))
    print('left with %d/%d samples'%(len(input_seqs), len(data)))
    print()

    data = {
        'tokens': input_seqs, 
        'span_start_index': span_starts,
        'span_end_index': span_ends
    }
    df = pd.DataFrame(data)
    df.to_csv(args.inp_csv,index=None,encoding='utf-8')