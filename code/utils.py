import ipdb
import pandas as pd

def fill_I_in_BEO(label_list, dict_lbl2idx):
    new_label_list = []
    span_ongoing = 0
    for idx, label_idx in enumerate(label_list):
        fill_value = False
        if label_idx == dict_lbl2idx['B']:
            # remember the starting of span
            if span_ongoing == 0:
                span_ongoing += 1
        elif label_idx == dict_lbl2idx['E']:
            # mark the ending of span
            if span_ongoing != 0:
                span_ongoing -= 1
            # else: erroneous prediction
        elif label_idx == dict_lbl2idx['O']:
            if span_ongoing != 0:
                fill_value = True
        else:
            raise ValueError('found %s (!=0,1,2) label-id in BEO encoding'%(label_idx))

        if fill_value:
            # 3 # B: 2, E: 1, O: 0
            new_label_list.append(3)
        else:
            new_label_list.append(label_idx)
        
        assert span_ongoing >= 0, ipdb.set_trace()
    return new_label_list