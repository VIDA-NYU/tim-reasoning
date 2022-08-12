import csv
import glob
import re
from tqdm import tqdm
import ast
from nltk import word_tokenize
import unicodedata
import argparse
import numpy as np 

def normalize(text):
    return unicodedata.normalize('NFD', text).lower()

## load recipe corpus 
files_list = glob.glob('../../multimodal-aligned-recipe-corpus/multi-step-breakdown/val/*/*')

## load all verbs and nouns in the epic kitchen dataset
verb_dict = dict()
noun_dict = dict()

with open('../../epic/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv', 'r') as ifile:
    reader = csv.reader(ifile, delimiter=',')
    for row in tqdm(reader):
        if row[0] == 'id':
            continue
        verb = row[1]
        verb_list = ast.literal_eval(row[2])
        category = row[3]
        for ele in verb_list:
            if ele not in verb_dict:
                verb_dict[ele] = category


with open('../../epic/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv', 'r') as ifile:
    reader = csv.reader(ifile, delimiter=',')
    for row in tqdm(reader):
        if row[0] == 'id':
            continue

        noun = row[1]
        noun_list = ast.literal_eval(row[2])
        category = row[3]
        for ele in noun_list:
            if ele not in noun_dict:
                noun_dict[ele] = category



sent_list = []
for file in files_list:
    with open(file) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            if row[0] == 'Source,Target':
                continue

            if row[0][:5] == '<http':
                continue
            if row[0][0] == ',':
                sent = row[0][1:]
                sent_list.append(sent)
            

            elif len(row) == 1:
                sent1 = ''
                sent2 = ''
                #print(len(row))
                sents = row[0].split(', ')
                for ii, sent in enumerate(sents):
                    if ',' not in sent:
                        sent1 += sent
                        sent1 += ', '
                    else:
                        #print(len(sent.split(',')))
                        if len(sent.split(',')) != 2:
                            break
                        assert len(sent.split(',')) == 2
                        sent1 += sent.split(',')[0]
                        sent2 += sent.split(',')[1]
                        sent2 += ', '.join(sents[(ii +1):])
                        break
                if sent1 != '':
                    sent_list.append(sent1)

                if sent2 != '':
                    sent_list.append(sent2)
print(len(sent_list))
## sent_list includes all extracted sentences 

ct = 0 
for sent in sent_list:
    ## TODO: probably use lemma of each word
    tokens = word_tokenize(normalize(sent))
    verbs = []
    nouns = []
    for vb in verb_dict:
        if vb in tokens:
            verbs.append(vb) 
    for noun in noun_dict:
        if noun in tokens:
            nouns.append(noun)
    
    if len(verbs) > 0 and len(nouns) > 0:
        ct += 1

    ##### Random sample positives and negatives