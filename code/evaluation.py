import argparse
import ipdb
import numpy as np
from utils import fill_I_in_BEO
from main import get_io_data
from metrics import compute_metrics

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv", type=str)
    parser.add_argument("--pred_csv", type=str)
    parser.add_argument("--encoding", type=str, default='io')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arg_parser()
    encoding = args.encoding

    gold_seqs, gold_labels = get_io_data(
        args.gold_csv,
        encoding,
    )
    pred_seqs, pred_labels = get_io_data(
        args.pred_csv,
        encoding,
    )

    dict_lbl2idx, dict_idx2lbl = {}, {}
    if encoding == 'bio':
        dict_lbl2idx = {'B': 2, 'I': 1, 'O': 0}
        dict_idx2lbl = {2:'B', 1:'I', 0:'O'}
    elif encoding == 'beio':
        dict_lbl2idx = {'E': 3, 'B': 2, 'I': 1, 'O': 0}
        dict_idx2lbl = {3:'E', 2:'B', 1:'I', 0:'O'}    
    elif encoding == 'io':
        dict_lbl2idx = {'I': 1, 'O': 0}
        dict_idx2lbl = {1:'I', 0:'O'}
    elif encoding == 'beo':
        dict_lbl2idx = {'B': 2, 'E': 1, 'O': 0}
        dict_idx2lbl = {2:'B', 1:'E', 0:'O'}
    else:
        raise NotImplementedError('unknown encoding: %s'%(encoding))

    assert len(gold_seqs) == len(pred_seqs)
    assert len(gold_labels) == len(pred_labels)

    for idx, (gold_label, pred_label) in enumerate(zip(gold_labels, pred_labels)):
        gold_label = [dict_lbl2idx[label] for label in gold_label]
        pred_label = [dict_lbl2idx[label] for label in pred_label]

        # Convert predictions to IO to have consistent comparison among BIO, BEIO and IO encodings.
        if args.encoding == 'beo':
            gold_label = fill_I_in_BEO(gold_label, dict_lbl2idx)
            pred_label = fill_I_in_BEO(pred_label, dict_lbl2idx)
        gold_label = (np.array(gold_label) >= 1).astype(np.int16).tolist()
        pred_label = (np.array(pred_label) >= 1).astype(np.int16).tolist()

        gold_labels[idx] = gold_label
        pred_labels[idx] = pred_label

    test_metrics = compute_metrics(
        gold_labels, pred_labels, 
        args.encoding, dict_lbl2idx, dict_idx2lbl
    )
    print(test_metrics)
