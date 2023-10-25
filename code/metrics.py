import numpy as np
import ipdb
from sklearn.metrics import precision_score, recall_score, f1_score

def get_token_list(labels):
    return [i for i in range(len(labels)) if labels[i]!=0]

def f1(predictions_bio, gold_bio):
    gold = get_token_list(gold_bio)
    predictions = get_token_list(predictions_bio)
    
    if len(gold) == 0:
        return 1 if len(predictions)==0 else 0
    nom = 2*len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions))+len(set(gold))
    return nom/denom


def get_dice(ground, preds):
    tot = 0
    for i in range(len(preds)):
        tot += f1(ground[i], preds[i])
    return tot/len(preds)

def get_hard_f1(ground, preds, micro=False):
    if micro:
        return np.mean([f1_score(l, p, average='micro', zero_division=0) for l, p in list(zip(ground, preds))])
    return np.mean([f1_score(l, p, average='macro', zero_division=0) for l, p in list(zip(ground, preds))])


def get_hard_recall(ground, preds, micro=False):
    if micro:
        return np.mean([recall_score(l, p, average='micro', zero_division=0) for l, p in list(zip(ground, preds))])    
    return np.mean([recall_score(l, p, average='macro', zero_division=0) for l, p in list(zip(ground, preds))])


def get_hard_precision(ground, preds, micro=False):
    if micro:
        return np.mean([precision_score(l, p, average='micro', zero_division=0) for l, p in list(zip(ground, preds))])    
    return np.mean([precision_score(l, p, average='macro', zero_division=0) for l, p in list(zip(ground, preds))])


def get_hard_recall_classwise(ground, preds):
    return np.mean([recall_score(l, p, labels=[0,1,2,3], average=None, zero_division=0)  for l, p in list(zip(ground, preds))], axis=0)


def get_hard_precision_classwise(ground, preds):
    return np.mean([precision_score(l, p, labels=[0,1,2,3], average=None, zero_division=0) for l, p in list(zip(ground, preds))], axis=0)


def get_hard_f1_classwise(ground, preds):
    return np.mean([f1_score(l, p, labels=[0,1,2,3], average=None, zero_division=0) for l, p in list(zip(ground, preds))], axis=0)


def compute_metrics(ground, 
                    preds,
                    encoding, 
                    dict_lbl2idx, 
                    dict_idx2lbl):
    # span-level scores
    span_p, span_r, span_f1 = [], [], []
    for (gold_labels, pred_labels) in zip(ground, preds):
        assert len(gold_labels) == len(pred_labels)
        gold_labels, pred_labels = np.array(gold_labels).astype(np.bool), np.array(pred_labels).astype(np.bool)
        intersection = np.sum(gold_labels * pred_labels)
        precision = intersection / np.sum(pred_labels) if np.sum(pred_labels) != 0 else 0
        recall = intersection / np.sum(gold_labels) if np.sum(gold_labels) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        
        span_p.append(precision)
        span_r.append(recall)
        span_f1.append(f1)
    
    span_scores = {
        'span_f1': round(100 * np.mean(span_f1), 2),
        'span_p': round(100 * np.mean(span_p), 2),
        'span_r': round(100 * np.mean(span_r), 2),
    }

    # token-level scores
    token_scores = {
        "token_f1": round(100 * get_hard_f1(ground, preds, micro=True), 2),
        "token_p": round(100 * get_hard_precision(ground, preds, micro=True), 2),
        "token_r": round(100 * get_hard_recall(ground, preds, micro=True), 2),
    }

    # f1 score for every tag - there are max 4 labels: O, I, E, B
    tag_scores = {}
    hard_f1_classwise=[round(v, 4) for v in get_hard_f1_classwise(ground, preds)]

    # Encoding set as IO for consistent comparison of different encodings
    # comment the below line to evaluate using the given encoding scheme
    encoding = 'io'
    dict_lbl2idx = {
        'O': 0,
        'I': 1,
    }

    if encoding == 'bio':
        f1_B = hard_f1_classwise[dict_lbl2idx['B']]
        f1_I = hard_f1_classwise[dict_lbl2idx['I']]
        f1_O = hard_f1_classwise[dict_lbl2idx['O']]
        tag_scores = {
            "F1_B": round(100 * f1_B, 2),
            "F1_I": round(100 * f1_I, 2),
            "F1_O": round(100 * f1_O, 2),
            "DSC": round(100 * get_dice(ground, preds), 2)
        }
    elif encoding == 'beio':
        f1_E = hard_f1_classwise[dict_lbl2idx['E']]
        f1_B = hard_f1_classwise[dict_lbl2idx['B']]
        f1_I = hard_f1_classwise[dict_lbl2idx['I']]
        f1_O = hard_f1_classwise[dict_lbl2idx['O']]
        tag_scores = {
            "F1_E": round(100 * f1_E, 3),
            "F1_B": round(100 * f1_B, 2),
            "F1_I": round(100 * f1_I, 2),
            "F1_O": round(100 * f1_O, 2),
            "DSC": round(100 * get_dice(ground, preds), 2)
        }
    elif encoding == 'io':
        f1_I = hard_f1_classwise[dict_lbl2idx['I']]
        f1_O = hard_f1_classwise[dict_lbl2idx['O']]
        tag_scores = {
            "F1_I": round(100 * f1_I, 2),
            "F1_O": round(100 * f1_O, 2),
            "DSC": round(100 * get_dice(ground, preds), 2)
        }
    elif encoding == 'beo':
        f1_E = hard_f1_classwise[dict_lbl2idx['E']]
        f1_B = hard_f1_classwise[dict_lbl2idx['B']]
        f1_O = hard_f1_classwise[dict_lbl2idx['O']]
        tag_scores = {
            "F1_E": round(100 * f1_E, 3),
            "F1_B": round(100 * f1_B, 2),
            "F1_O": round(100 * f1_O, 2),
            "DSC": round(100 * get_dice(ground, preds), 2)
        }
    else:
        raise ValueError('Wrong encoding passed: %s'%(encoding))

    # return all scores
    metrics = {}
    for (k,v) in span_scores.items():
        metrics[k] = v
    for (k,v) in token_scores.items():
        metrics[k] = v
    for (k,v) in tag_scores.items():
        metrics[k] = v
    return metrics