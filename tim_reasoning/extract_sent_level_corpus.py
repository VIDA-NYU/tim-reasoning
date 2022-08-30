import csv
import glob
import re
from tqdm import tqdm
import ast
from nltk import word_tokenize
import unicodedata
import argparse
import numpy as np 
import random
import json

def normalize(text):
    return unicodedata.normalize('NFD', text).lower()

## load recipe corpus 
files_list = glob.glob('../../multimodal-aligned-recipe-corpus/textual-visual-paraphrases/test/*/*')


sent_list = []
instances = []
ct = 0 


for file in files_list:
    last_name = file.split('/')[-1]
    if last_name == 'text_to_video_paraphrases.csv':
        continue
    sent_dict = dict()
    with open(file) as ifile:
        
        reader = csv.reader(ifile, delimiter=',')
        for row in reader:
            if row[0] == 'Source' and row[1] == 'Target':
                continue
            if row[0] == '':
                continue
            if row[2] == '':
                continue
            score = float(row[2])
            sent1 = row[0]
            sent2 = row[1]
            if score > 0.8:
                if row[0] not in sent_dict:
                    sent_dict[sent1] = {'pos': [], 'neg': []}
                sent_dict[sent1]['pos'].append(sent2)
    with open(file) as ifile:    
        reader = csv.reader(ifile, delimiter=',')
        
        for row in reader:
            if row[0] == 'Source' and row[1] == 'Target':
                continue
            if row[0] == '':
                continue
            if row[2] == '':
                continue
            score = float(row[2])
            sent1 = row[0]
            sent2 = row[1]
            for sent in sent_dict:
                if sent != sent1:
                    sent_dict[sent]['neg'].append(sent2)
    
    for sent in sent_dict:
        if len(sent_dict[sent]['neg']) < 3:
            continue
        for sent2 in sent_dict[sent]['pos'][:5]:
            instances.append({'id': ct, 'sent1':sent, 'sent2':sent2, 'label': 1})
            ct += 1
        
        num_neg = min(len(sent_dict[sent]['pos']), 5)
        neg_sents = random.choices(sent_dict[sent]['neg'], k=num_neg)	
        for sent2 in neg_sents:
            instances.append({'id': ct, 'sent1':sent, 'sent2':sent2, 'label': 0})
            ct += 1
        
            

print(len(instances))
print(ct)

with open('../data/description_data_test.json', 'w', encoding='utf-8') as f:
    json.dump(instances, f, indent=2)

