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
files_list = glob.glob('../../../multimodal-aligned-recipe-corpus/multi-step-breakdown/train/*/*')

## load all verbs and nouns in the epic kitchen dataset
verb_dict = dict()
noun_dict = dict()

with open('../../../epic/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv', 'r') as ifile:
    reader = csv.reader(ifile, delimiter=',')
    for row in tqdm(reader):
        if row[0] == 'id':
            continue
        verb = row[1]
        verb_list = ast.literal_eval(row[2])
        category = row[3]
        for ele in verb_list:
            if ':' in ele:
                continue
            if ele not in verb_dict:
                verb_dict[ele] = category


with open('../../../epic/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv', 'r') as ifile:
    reader = csv.reader(ifile, delimiter=',')
    for row in tqdm(reader):
        if row[0] == 'id':
            continue

        noun = row[1]
        noun_list = ast.literal_eval(row[2])
        category = row[3]
        for ele in noun_list:
            if ':' in ele:
                continue
            if ele not in noun_dict:
                noun_dict[ele] = category


noun_verb_list = []
### extract all verb and noun combinations 
with open('../../../epic/EPIC_100_train.csv', 'r') as ifile:
    reader = csv.reader(ifile, delimiter=',')
    for row in tqdm(reader):
        if row[0] == 'narration_id':
            continue
        verb = row[9]
        noun_list = ast.literal_eval(row[13])
        for noun in noun_list:
            if ':' in noun:
                noun = noun.split(':')[1]
                #print(noun)
                #input()
            pair_string = noun + '###' + verb
            if pair_string not in noun_verb_list:
                noun_verb_list.append(pair_string)

with open('../../../epic/EPIC_100_validation.csv', 'r') as ifile:
    reader = csv.reader(ifile, delimiter=',')
    for row in tqdm(reader):
        if row[0] == 'narration_id':
            continue
        verb = row[9]
        noun_list = ast.literal_eval(row[13])
        for noun in noun_list:
            if ':' in noun:
                noun = noun.split(':')[1]
            pair_string = noun + '###' + verb
            if pair_string not in noun_verb_list:
                noun_verb_list.append(pair_string)



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
#print(noun_verb_list)
instances = []
ct = 0 
count = 0
for sent in tqdm(sent_list):
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
    
    all_pairs = []
    for vb in verbs:
        for noun in nouns:
            #if len(vb) == 1 or len(noun) == 1:
            #    continue
            pair = noun + '###' + vb
            if pair in noun_verb_list and pair not in all_pairs:
                all_pairs.append(pair)
    #print(verbs, nouns)
    #print(all_pairs)
    #input()
    
    if len(all_pairs) > 0:
        ct += 1
    else:
        continue
    ##### Random sample positives and negatives
    for pair in all_pairs:
        noun, vb = pair.split('###')
        ### positives 
        random_num = random.random()
        if random_num < 0.5:             
            instances.append({'id': count, 'step': sent, 'action_vb': vb, 'action_noun': noun, 'label': 1})
            count += 1
        elif random_num < 0.75:
            cand_vbs = []
            for c_v in verb_dict:
                if c_v == vb:
                    continue
                if verb_dict[c_v] == verb_dict[vb]:
                    cand_vbs.append(c_v)
                if len(cand_vbs) == 0:
                    continue
            new_vb = random.choice(cand_vbs)
            instances.append({'id': count, 'step': sent, 'action_vb': new_vb, 'action_noun': noun, 'label': 1})
            count += 1
        else:
            cand_nouns = []
            for c_v in noun_dict:
                if c_v == noun:
                    continue
                if noun_dict[c_v] == noun_dict[noun]:
                    cand_nouns.append(c_v)
                if len(cand_nouns) == 0:
                    continue
            new_noun = random.choice(cand_nouns)
            instances.append({'id': count, 'step': sent, 'action_vb': vb, 'action_noun': new_noun, 'label': 1})
            count += 1

        ### negatives 
        random_neg = random.random()
        cand_vbs = []
        cand_nouns = []
        for c_v in verb_dict:
            if c_v == vb:
                continue
            if verb_dict[c_v] != verb_dict[vb]:
                cand_vbs.append(c_v)
        for c_v in noun_dict:
            if c_v == noun:
                continue
            if noun_dict[c_v] != noun_dict[noun]:
                cand_nouns.append(c_v)

        new_vb = random.choice(cand_vbs)
        new_noun = random.choice(cand_vbs)

        if random_neg < 0.3:
            instances.append({'id': count, 'step': sent, 'action_vb': new_vb, 'action_noun': noun, 'label': 0})
            count += 1
        elif random_neg < 0.6:
            instances.append({'id': count, 'step': sent, 'action_vb': vb, 'action_noun': new_noun, 'label': 0})
            count += 1
        else:
            instances.append({'id': count, 'step': sent, 'action_vb': new_vb, 'action_noun': new_noun, 'label': 0})
            count += 1
print(len(instances))

with open('../../data/action_data_train.json', 'w', encoding='utf-8') as f:
    json.dump(instances, f, indent=2)

