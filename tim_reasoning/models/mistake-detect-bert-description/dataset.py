from tqdm import tqdm
from random import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from enum import IntEnum
import os
import collections
import random

import json
import pickle



def load_data_train(data_file, isTrain):
    with open(data_file, 'r') as f:
        data = json.load(f)

    #data = list()
    #print(len(dataset))
    #if isTrain and 'dev' in data_file:
    #    dataset = dataset[:30000]
    

    #for q in dataset:
    #    data.append(q)

    return data

def truncate_input_sequence(tokens_a, tokens_b, max_num_tokens):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > 0 else tokens_b
        # assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def encode_sequence(question, passage, max_seq_len, tokenizer):
    #span_data.append({'qid': qid, 'question': q['question'], 'passage': context_tokens, 'evidence': list(), 'answer': q['answer'], 'spans':[]})

    seqA = tokenizer.tokenize(question)
    #seqB = tokenizer.tokenize(textB)
    #if p:
    #    print (seqA, seqB)
    #truncate_input_sequence(seqA, seqB, max_seq_len-3)
    seqA = ["[CLS]"] + seqA + ["[SEP]"]

    seqB = tokenizer.tokenize(passage) + ["[SEP]"]
    truncate_input_sequence(seqA, seqB, max_seq_len)

    input_tokens = seqA + seqB

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    sequence_ids = [0]*len(seqA) + [1]*len(seqB)
    input_mask = [1]*len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        sequence_ids.append(0)
        input_mask.append(0)

    return (map_to_torch(input_ids), map_to_torch(input_mask), map_to_torch(sequence_ids))


def batch_transform_bert(inst, bert_max_len, bert_tokenizer):
    qid = inst['id']
    question = inst['sent1']
    label = inst['label']
    
    passage = inst['sent2']

    ids, mask, type_ids = encode_sequence(question, passage, bert_max_len, bert_tokenizer)
    
    return (ids, mask, type_ids, label, qid)


def map_to_torch_float(encoding):
    encoding = torch.FloatTensor(encoding)  
    encoding.requires_grad_(False)
    return encoding


def map_to_torch(encoding):
    encoding = torch.LongTensor(encoding)  
    encoding.requires_grad_(False)
    return encoding




class HotpotSpanDataset(Dataset):
    def __init__(self, filename, config_model, isTrain=False, bert_tokenizer=None):
        self.config_model = config_model
        self.bert_tokenizer = bert_tokenizer
        self.istrain = isTrain
        self.data = load_data_train(filename, isTrain)
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len
        inst = self.data[index]

        ids, mask, type_ids, label, qid = batch_transform_bert(inst, self.config_model['bert_max_len'], self.bert_tokenizer)
        
        return [ids, mask, type_ids, label, qid, inst['sent1']]
 