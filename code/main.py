import argparse
import os
import json
import logging
import numpy as np
import pandas as pd
import ast
import ipdb
from tqdm import tqdm
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
from sklearn.metrics import precision_score, recall_score, f1_score
from metrics import compute_metrics
from utils import fill_I_in_BEO
import warnings

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_train", type=str, default='../data/train-en.csv')
    parser.add_argument("--path_dev", type=str, default='../data/dev-en.csv')
    parser.add_argument("--path_test",  type=str, default='../data/test-en.csv')
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--checkdir", type=str, default="../ckpts")
    parser.add_argument("--logdir", type=str, default="../logdir")
    parser.add_argument("--plm", type=str, default='microsoft/mdeberta-v3-base')
    parser.add_argument("--name", type=str, default='mono_en_mdeberta')
    parser.add_argument("--encoding", type=str, default='io')
    parser.add_argument("--weights", default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--plm_lr", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50, help='maximum number of training epochs')
    parser.add_argument("--patience", type=int, default=7, help='# patience epochs for early stopping')
    parser.add_argument("--gpu", default=42, type=int)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    return args

def get_io_data(
        csv_path, 
        encoding,
        test=False
    ):
    data = pd.read_csv(csv_path)

    seqs = []
    # if not test:
    labels = []

    count_overlapping_spans, count_spans = 0, 0
    for i in range(len(data)):
        seq = ast.literal_eval((data['tokens'][i]))
        seqs.append(seq)

        # if not test:
        label = ['O'] * len(seq)
        span_starts = json.loads(data['span_start_index'][i])
        span_ends = json.loads(data['span_end_index'][i])
        assert len(span_starts) == len(span_ends), ipdb.set_trace()

        for (span_s, span_e) in zip(span_starts, span_ends):
            count_spans += 1
            if span_s >= len(seq):
                span_s = len(seq) - 1
            if span_e >= len(seq):
                span_e = len(seq) - 1

            # mark the span starting from span_s to span_e (included)
            for idx in range(span_s, span_e+1):

                if label[idx] != 'O':
                    count_overlapping_spans += 1

                if encoding == 'bio':
                    label[idx] = 'B' if idx == span_s else 'I'
                elif encoding == 'io':
                    label[idx] = 'I'
                elif encoding == 'beo':
                    if idx == span_s:
                        label[idx] = 'B'
                    elif idx == span_e:
                        label[idx] = 'E'
                    # else: label[idx] need not be 'O' due to presence of overlapping spans
                elif encoding == 'beio':
                    if idx == span_s:
                        label[idx] = 'B'
                    elif idx == span_e:
                        label[idx] = 'E'
                    else:
                        label[idx] = 'I'
                else:
                    raise NotImplementedError('unknown encoding: %s'%(encoding))

        labels.append(label)

    print('Found %d/%d overlapping spans in %s'%(
        count_overlapping_spans,
        count_spans,
        csv_path)
    )

    return (seqs, labels)
    # return (seqs, None)

class myDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_seq_len, dict_lbl2idx, label_all_tokens=True, test=False):
        super(myDataset, self).__init__()
        self.test = test
        self.examples = examples
        self.seqs = examples[0]

        # tokenize each word manually since there are weird words preseq like '\ufeff' or '' which tokenize into nothing
        seqs_tokenized = {
            'input_ids': [],
            'attention_mask': [],
            'word_ids': []
        }
        cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        for (idx, seq) in enumerate(self.seqs):
            input_ids, attention_mask, word_ids = [], [], []
            for idx, word in enumerate(seq):
                token_ids = tokenizer(word)['input_ids'][1:-1] # leave cls and sep

                if len(token_ids) == 0:
                    token_ids = [pad_id]

                if (len(input_ids) + len(token_ids) > max_seq_len-2): # not test and 
                    # print('truncating')
                    break

                input_ids += token_ids
                attention_mask += ([1] * len(token_ids))
                word_ids += ([idx] * len(token_ids))

            input_ids = [cls_id] + input_ids + [sep_id]
            attention_mask = [1] + attention_mask + [1]
            word_ids = [None] + word_ids + [None]

            input_ids = input_ids + [pad_id] * (max_seq_len - len(input_ids))
            attention_mask = attention_mask + [0] * (max_seq_len - len(attention_mask))
            word_ids = word_ids + [None] * (max_seq_len - len(word_ids))

            assert len(input_ids) == len(attention_mask) and len(word_ids) == len(attention_mask)
            if len(input_ids) > max_seq_len:
                print('found seq with len > max_seq_len')
                ipdb.set_trace()

            seqs_tokenized['input_ids'].append(input_ids)
            seqs_tokenized['attention_mask'].append(attention_mask)
            seqs_tokenized['word_ids'].append(word_ids)

        seqs_tokenized['input_ids'] = torch.tensor(seqs_tokenized['input_ids'])
        seqs_tokenized['attention_mask'] = torch.tensor(seqs_tokenized['attention_mask'])

        self.seqs_tokenized = seqs_tokenized
        self.word_ids = seqs_tokenized['word_ids']
        self.input_ids = seqs_tokenized['input_ids']
        self.attention_mask = seqs_tokenized['attention_mask']
        assert len(self.input_ids) == len(self.attention_mask)

        labels = examples[1]
        labels_tokenized = []
        for idx, seq_label in enumerate(labels):
            # word_ids = seqs_tokenized.word_ids(batch_index=idx)
            word_ids = self.word_ids[idx]
            previous_word_idx = None
            seq_label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
                if word_idx is None: # tokenizer.special_tokens_map
                    seq_label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    text_label = seq_label[word_idx]
                    seq_label_ids.append(dict_lbl2idx[text_label])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                elif word_idx == previous_word_idx:
                    text_label = seq_label[word_idx]
                    seq_label_ids.append(dict_lbl2idx[text_label] if label_all_tokens else -100)
                previous_word_idx = word_idx
            
            assert len(word_ids) == len(seq_label_ids)
            labels_tokenized.append(seq_label_ids)
        
        self.y = torch.tensor(labels_tokenized)
        assert len(self.y) == len(self.input_ids)
    
    def __getitem__(self, idx):
        if self.test:
            return self.input_ids[idx], self.attention_mask[idx]
        else:
            return self.input_ids[idx], self.attention_mask[idx], self.y[idx]
    
    def __len__(self):
        return len(self.input_ids)

class myModel(nn.Module):
    def __init__(self, plm, hidden_dim=768, dropout=0.3, n_labels=3):
        super(myModel, self).__init__()
        self.plm = plm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, inp):
        out = self.plm(**inp)
        if len(out) == 2:
            sequence_output, pooled_output = out[0], out[1]
        elif len(out) == 1:
            sequence_output = out[0]
        else:
            raise NotImplementedError('Output of pretrained lm is of length %d'%(len(out)))
        logits = self.classifier(sequence_output) # [bs, seq_len, n_classes]
        preds = logits.argmax(-1)
        return logits, preds

def train(model, train_loader, device, criterion, optimizers, schedulers, ep):
    model.train()
    train_loss, y_true, y_pred = [], [], []
    pbar = tqdm(train_loader)
    for batch_idx, (inp_id, attn, y) in enumerate(pbar):
        batch = {
            "input_ids": inp_id.to(device),
            "attention_mask": attn.to(device),
        }
        y = y.to(device)
        
        for optim in optimizers.values():
            optim.zero_grad()

        logits, preds = model(batch)
        loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
    
        loss.backward()
        for optim in optimizers.values():
            optim.step()
        for scheduler in schedulers.values():
            scheduler.step()

        y_true.extend(y.view(-1).cpu().detach().numpy().tolist())
        y_pred.extend(preds.view(-1).cpu().detach().tolist())

        train_loss.append(loss.item())

    new_y_true, new_y_pred = [], []
    for (y1, y2) in zip(y_true, y_pred):
        if y1>=1: # exclude O labels
            new_y_true.append(y1)
            new_y_pred.append(y2)
    y_true, y_pred = new_y_true, new_y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")
        recall = recall_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")
        f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")

    return np.mean(train_loss), precision, recall, f1

def eval(model, dev_loader, device, criterion, ep):
    model.eval() 
    dev_loss, y_true, y_pred = [], [], []
    with torch.no_grad():
        pbar = tqdm(dev_loader)
        for batch_idx, (inp_id, attn, y) in enumerate(pbar):
            batch = {
                "input_ids": inp_id.to(device),
                "attention_mask": attn.to(device),
            }
            y = y.to(device)

            logits, preds = model(batch)
            loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))

            y_true.extend(y.view(-1).cpu().detach().numpy().tolist())
            y_pred.extend(preds.view(-1).cpu().detach().tolist())

            dev_loss.append(loss.item())

    new_y_true, new_y_pred = [], []
    for (y1, y2) in zip(y_true, y_pred):
        if y1>=1:
            new_y_true.append(y1)
            new_y_pred.append(y2)
    y_true, y_pred = new_y_true, new_y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")
        recall = recall_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")
        f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")

    return np.mean(dev_loss), precision, recall, f1

def test(model, test_loader, device):
    model.eval() 
    all_preds = []
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch_idx, (inp_id, attn) in enumerate(pbar):
            batch = {
                "input_ids": inp_id.to(device),
                "attention_mask": attn.to(device),
            }

            logits, preds = model(batch)
            preds = preds.cpu().detach().tolist()
            all_preds.extend(preds)

    return all_preds

if __name__ == "__main__":
    args = get_arg_parser()
    pl.seed_everything(args.seed)
    if args.gpu == -1:
        device = 'cpu'
    elif args.gpu == 42:
        device = 'cuda'
    else:
        device = 'cuda:%d'%(args.gpu)

    max_seq_len, max_epochs, batch_size = args.max_seq_len, args.epochs, args.bs
    if not os.path.exists(args.checkdir): os.mkdir(args.checkdir)
    if not os.path.exists(args.logdir): os.mkdir(args.logdir)

    # Create and configure logger
    log_file_path = os.path.join(args.logdir, args.name + '.log')
    logging.basicConfig(filename=log_file_path,
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(args)

    # mapping between encoding and index
    encoding = args.encoding
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

    # model
    plm = AutoModel.from_pretrained(args.plm)
    plm_tokenizer = AutoTokenizer.from_pretrained(args.plm)

    num_labels = 1
    if encoding == 'bio' or encoding == 'beo':
        num_labels = 3
    elif encoding == 'beio':
        num_labels = 4
    elif encoding == 'io':
        num_labels = 2
    else:
        raise NotImplementedError('unknown encoding: %s'%(encoding))
    model = myModel(
        plm=plm, 
        hidden_dim=plm.config.hidden_size, 
        dropout=0.3, 
        n_labels=num_labels
    ).to(device)

    # dataset
    kwargs= {}

    if args.train:
        train_seqs, train_labels = get_io_data(
            args.path_train,
            encoding,
        )
        dev_seqs, dev_labels = get_io_data(
            args.path_dev,
            encoding,
        )
        logger.info("#train = %d, #dev = %d"%(len(train_seqs), len(dev_labels)))

        trainset = myDataset((train_seqs, train_labels), plm_tokenizer, max_seq_len, dict_lbl2idx)
        devset = myDataset((dev_seqs, dev_labels), plm_tokenizer, max_seq_len, dict_lbl2idx)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        dev_loader = torch.utils.data.DataLoader(devset, batch_size=batch_size, shuffle=False, **kwargs)
        logger.info("# train samples = %d, # dev samples = %d"%(len(trainset), len(devset)))

    else:
        test_seqs, test_labels = get_io_data(
            args.path_test,
            encoding,
            test=False,
        )
        logger.info("#test = %d"%(len(test_seqs)))

        testset = myDataset((test_seqs, test_labels), plm_tokenizer, max_seq_len, dict_lbl2idx, test=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
        logger.info("# test samples = %d"%(len(testset)))

    if args.weights:
        logger.info("loading model from %s"%(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    if args.train:
        optimizers, schedulers = {}, {}
        optimizers["plm_optimizer"] = torch.optim.Adam(
            model.plm.parameters(), lr=args.plm_lr
        )
        schedulers["plm_scheduler"] = transformers.get_linear_schedule_with_warmup(
            optimizers["plm_optimizer"],
            0, len(trainset) * max_epochs,
        )

        optimizers["general_optimizer"] = torch.optim.Adam(
            model.classifier.parameters(), lr=args.lr
        )
        schedulers["general_scheduler"] = transformers.get_linear_schedule_with_warmup(
            optimizers["general_optimizer"],
            0, len(trainset) * max_epochs
        )

        logger.info('Optimizers loaded. Starting training...')
        best_f1, count_for_ea = 0, 0
        for ep in range(max_epochs):
            train_l, train_p, train_r, train_f = train(model, train_loader, device, criterion, optimizers, schedulers, ep)
            dev_l, dev_p, dev_r, dev_f = eval(model, dev_loader, device, criterion, ep)

            logger.info('epoch:%d (loss, precision, recall, f1) train=(%.2f, %.2f, %.2f, %.2f) dev=(%.2f, %.2f, %.2f, %.2f)'\
                %(ep, train_l, train_p, train_r, train_f, dev_l, dev_p, dev_r, dev_f))

            if (ep <= 5) or (dev_f > best_f1):
                best_f1 = dev_f
                logger.info(f"saving model")
                torch.save(model.state_dict(), os.path.join(args.checkdir, args.name+'.pt'))
                count_for_ea = 0
            else:
                count_for_ea += 1

                if count_for_ea == args.patience:
                    logger.info("patience reached! early stopping... with best dev f1 = %.2f"%(best_f1))
                    break

    else:
        subword_preds = test(model, test_loader, device)
        subword_labels = testset.y.cpu().detach().tolist() # label_tokenized
        assert len(subword_preds)==len(subword_labels), ipdb.set_trace()

        word_preds, word_labels = [], []
        for idx, (label, pred) in enumerate(zip(subword_labels, subword_preds)):
            word_ids = testset.word_ids[idx]
            label, pred, word_ids = label[1:], pred[1:], word_ids[1:]
            assert len(word_ids) == len(pred) and len(pred) == len(label), ipdb.set_trace()

            # get word level labels (in gold and pred) from subword piece level
            word_pred, word_label = [], []
            previous_word_idx = None
            for subword_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    break
                elif word_idx != previous_word_idx:
                    word_label.append(label[subword_idx])
                    word_pred.append(pred[subword_idx])
                
                previous_word_idx = word_idx
            
            if len(word_pred) != len(word_label):
                print('possibly found where white space is causing tokenization mismap issue')
                seq = testset.seqs[idx]
                seq = [word for word in seq if len(word)!=0]
                assert len(word_label)==len(seq), ipdb.set_trace()

            if len(word_label) != len(testset.seqs[idx]):
                word_label = []
                for tag in testset.examples[1][idx]:
                    if tag == 'O':
                        word_label.append(0)
                    else:
                        word_label.append(1)
                word_pred = word_pred + [0] * (len(word_label) - len(word_pred))

            # Convert predictions to IO to have consistent comparison among BIO, BEIO and IO encodings.
            if args.encoding == 'beo':
                word_label = fill_I_in_BEO(word_label, dict_lbl2idx)
                word_pred = fill_I_in_BEO(word_pred, dict_lbl2idx)
            word_label = (np.array(word_label) >= 1).astype(np.int16).tolist()
            word_pred = (np.array(word_pred) >= 1).astype(np.int16).tolist()
            
            word_labels.append(word_label) 
            word_preds.append(word_pred)

        test_metrics = compute_metrics(word_labels, word_preds, args.encoding, dict_lbl2idx, dict_idx2lbl)
        print(test_metrics)

        # # write the predictions (e.g., for error analysis)
        # # contiguous span
        # pred_texts, pred_span_s, pred_span_e = [], [], []
        # for (pred_text, pred_label) in zip(testset.seqs, word_preds):
        #     pred_texts.append(pred_text)
        #     if np.sum(pred_label)>=1:
        #         span_s = pred_label.index(1)
        #         span_e = len(pred_label) - pred_label[::-1].index(1) - 1
        #         pred_span_s.append([span_s])
        #         pred_span_e.append([span_e])
        #     else:
        #         pred_span_s.append([])
        #         pred_span_e.append([])
        
        # pred_df = pd.DataFrame({
        #     'tokens': pred_texts,
        #     'span_start_index': pred_span_s,
        #     'span_end_index': pred_span_e,
        # })
        # pred_df.to_csv('./pred-csv',index=None,encoding='utf-8')

        # # discontiguous span, which is used in evaluation
        # pred_texts, pred_claims = [], []
        # for (pred_text, pred_label) in zip(testset.seqs, word_preds):
        #     pred_texts.append(pred_text)
        #     claim = []
        #     for (token, token_label) in zip(pred_text, pred_label):
        #         if token_label == 1:
        #             claim += [token]
        #     pred_claims.append(claim)
        
        # pred_texts = [' '.join(text) for text in pred_texts]
        # pred_claims = [' '.join(claim) for claim in pred_claims]
        # pred_df = pd.DataFrame({
        #     'text': pred_texts,
        #     'claim': pred_claims
        # })
        # pred_df.to_csv('./pred-hi.csv',index=None,encoding='utf-8')
