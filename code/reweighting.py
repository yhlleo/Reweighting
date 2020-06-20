#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Yahui Liu <yahui.liu@unitn.it>

import os
import codecs
import numpy as np
import json
from collections import OrderedDict

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--src_file', help='/path/to/source_input')
parser.add_argument('--tgt_file', help='/path/to/target_input')
parser.add_argument('--src_output', help='/path/to/source_output')
parser.add_argument('--tgt_output', help='/path/to/target_output')
parser.add_argument('--wt_output', help='/path/to/weight_output')
parser.add_argument('--reweight_mode', default='FL', 
    help='[F, L, FL], F refers to frequency, L refers to length, and FL is the corresponding combination.')
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--coff0', default=0.33, type=float)
parser.add_argument('--coff1', default=0.33, type=float)
parser.add_argument('--bias', default=3.0, type=float)
args = parser.parse_args()

punctuations = {
    u'，', u'。', u'、', u'？', u'；',
    u'：', u'‘', u'’', u'“', u'”',
    u'.', u',', u'!', u'~', u'[',
    u']', u'{', u'}', u'|', u'-',
    u'+', u'*', u'/', u'#', u'￥',
    u'%', u'…', u'&', u'(', u')',
    u'（', u'）', u'—', u'【', u'】'
}

def qa_pairs(src_file, tar_file):
    qa_dict = OrderedDict()
    for query, response in zip(
        codecs.open(src_file, 'r', encoding='utf-8'),
        codecs.open(tar_file, 'r', encoding='utf-8')):
        
        query = query.strip()
        response = response.strip()
        
        if query not in qa_dict:
            qa_dict[query] = []
        qa_dict[query].append([response, len(response.split(' '))])
    return qa_dict

def remove_punc(sent, punc=punctuations):
    for pt in punctuations:
        if pt in sent:
            sent = sent.replace(pt, '')    
    return sent

def get_freqs(tar_file):
    res_freqs = OrderedDict()
    with codecs.open(tar_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            res = remove_punc(line.strip().replace(' ',''))

            if res not in res_freqs:
                res_freqs[res] = 0
            res_freqs[res] += 1
    return res_freqs

def average_len(src_file, tar_file):
    avg_src, count_src = 0.0, 0
    avg_tar, count_tar = 0.0, 0

    for src, tar in zip(
        codecs.open(src_file, 'r', encoding='utf-8'),
        codecs.open(tar_file, 'r', encoding='utf-8')):
        
        len_src = len(src.strip().replace('\n', '').split(' '))
        len_tar = len(tar.strip().replace('\n', '').split(' '))

        avg_src += len_src
        count_src += 1

        avg_tar += len_tar
        count_tar += 1
    avg_src /= float(count_src)
    avg_tar /= float(count_tar)

    return (avg_src, count_src), (avg_tar, count_tar)

def reweight_responses(qa_dict, 
                       res_freqs,
                       ncount, 
                       outputs,
                       alpha=0.5, 
                       coff0=0.33, 
                       bias=3.0, 
                       coff1=0.33, 
                       avg_len=10.24, 
                       mode='FLG'):
    assert mode in ['F', 'L', 'FL']
    fouts = [codecs.open(op_file, 'w', encoding='utf-8') for op_file in outputs]
    for k, v in qa_dict.items():
        freqs, lens = [], []
        for res in v:
            freqs.append(res_freqs[remove_punc(res[0].strip().replace(' ',''))])
            lens.append(res[1])

        ws = []
        for f, l in zip(freqs, lens):
            fw = 1.0 if f <= bias else np.exp(-coff0*(f-bias))
            lw = np.exp(-np.abs(l-avg_len)*coff1)

            if mode == 'FL':
                ws.append(alpha*fw + (1-alpha)*lw)
            if mode == 'F':
                ws.append(fw)
            if mode == 'L':
                ws.append(lw)

        max_ws = np.max(ws)
        ws = [w/max_ws for w in ws]

        for i in range(len(ws)):
            fouts[0].write(k+u'\n')
            fouts[1].write(v[i][0]+u'\n')
            fouts[2].write(u'%.4f'%(ws[i]) + u'\n')

    for f in fouts:
        f.close()

if __name__ == '__main__':
    src_file = args.src_file 
    tar_file = args.tgt_file 

    outputs = [
        args.src_output, 
        args.tgt_output, 
        args.wt_output] 

    src, tar = average_len(src_file, tar_file)
    print('average length: ': src, tar)

    qa_dict = qa_pairs(src_file, tar_file)
    print('QA dict done!')

    res_freqs = get_freqs(tar_file)
    print('Responses frequencies done!')

    reweight_responses(
        qa_dict,
        res_freqs,
        ncount,
        outputs,
        coff0=args.coff0,
        coff1=args.coff1,
        bias=args.bias,
        avg_len=tar[0],
        alpha=args.alpha,
        mode=args.reweight_mode)
    print('Finished!')
