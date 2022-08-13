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
import torch
from model import DeepQDSBertModel
from dataset import HotpotSpanDataset, batch_transform_bert, encode_sequence
from pytorch_transformers.tokenization_bert import BertTokenizer
import os


### TODO: replace it with actual files 

recipe = ["Place tortilla on cutting board.",
    "Use a butter knife to scoop nut butter from the jar.",
    "Spread nut butter onto tortilla, leaving 1/2-inch uncovered at the edges."
    #"Clean the knife by wiping with a paper towel.",
    "Use the knife to scoop jelly from the jar.",
    "Spread jelly over the nut butter."
    #"Clean the knife by wiping with a paper towel.",
    "Roll the tortilla from one end to the other into a log shape, about 1.5 inches thick.",
    #"Roll it tight enough to prevent gaps, but not so tight that the filling leaks.",
    "Secure the rolled tortilla by inserting 5 toothpicks about 1 inch apart.",
    "Trim the ends of the tortilla roll with the butter knife, leaving 1⁄2 inch margin between the last toothpick and the end of the roll. Discard ends.",
    "Slide floss under the tortilla, perpendicular to the length of the roll.",
    "Cross the two ends of the floss over the top of the tortilla roll.",
    #"Holding one end of the floss in each hand, pull the floss ends in opposite directions to slice."
    "Place the pinwheels on a plate."]

actions = ['Grab tortilla', 'Grab tortilla from plate', 'Place tortilla',
'Place tortilla on cutting board', 'Scoop nut butter', 'Scoop peanut butter',
'Scoop peanut butter from jar', 'Spread onto tortilla', 'Spread peanut butter',
'Spread nut butter onto tortilla', 'Scoop jelly', 'Scoop grape jelly from jar',
'Spread grape jelly', 'Spread jelly onto tortilla', 'Roll tortilla',
'Roll tortilla into wrap', 'Insert toothpicks',
'Insert toothpicks into tortilla', 'Trim tortilla', 'Trim tortilla ends',
'Cut tortilla ends', 'Slide floss under tortilla',
'Slide string under tortilla', 'Cross floss', 'Cross floss across tortilla',
'Pull string', 'Pull string to slice roll', 'Cross roll',
'Place pinwheels on plate', 'Place wraps on plate',  'Place mini wraps on plate']

#### Load BERT classifier, assuming only one GPU 

#logging.info("********* Load Mistake Detector ************")
parser = argparse.ArgumentParser("Hotpot Edge Ranking")
config = json.load(open('configs/config_bert.json', 'r', encoding="utf-8"))
args = parser.parse_args()
args.fp16 = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = DeepQDSBertModel(args, config, None)
model.network.to(device)
model.load(os.path.join('/home/zhaochen/postdoc/tim-reasoning-models/', "saved_models/model_finetuned_epoch_{}.pt".format(0)))
model.eval()

### Run Evaluation, step by step

curr_step = 0
executed = False
for ii in range(len(actions)):
     question = recipe[curr_step]
     passage = actions[ii]

     ids, mask, type_ids = encode_sequence(question, passage, 128, bert_tokenizer)
     ids = ids.unsqueeze(0).to(device)
     mask = mask.unsqueeze(0).to(device)
     type_ids = type_ids.unsqueeze(0).to(device)
     batch_eval_data = tuple([ids, mask, type_ids, None, None])

     output = model.network(batch_eval_data)
     pred = output.argmax(dim=1).data.cpu().numpy().tolist()

     ### State management
     ### TODO: after multiple mistakes, should be jump to later steps directly???
     ### TODO: discuss which steps it should compare to 
     print(ii)
     print(output.data.cpu().numpy().tolist())
     #input()
     if curr_step < len(recipe) - 1 and executed == True:
        question = recipe[curr_step + 1]
        ids, mask, type_ids = encode_sequence(question, passage, 128, bert_tokenizer)
        ids = ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        type_ids = type_ids.unsqueeze(0).to(device)
        batch_eval_data = tuple([ids, mask, type_ids, None, None])

        output1 = model.network(batch_eval_data)
        pred1 = output1.argmax(dim=1).data.cpu().numpy().tolist()
        if pred1[0] == 1 and output.data.cpu().numpy().tolist()[0][1] < output1.data.cpu().numpy().tolist()[0][1]:
            curr_step += 1
            executed = True
            print('Current recipe step:', curr_step, recipe[curr_step])
            print('Input action:', actions[ii])
            print('Is a mistake:', 'No')
            print('*********************')

        elif pred1[0] == 0 and pred[0] == 0:
            print('Current recipe step:', curr_step, recipe[curr_step])
            print('Input action:', actions[ii])
            print('Is a mistake:', 'Yes')
            print('*********************')
        else:
            #assert pred1[0] == 0 and pred[0] == 1
            print('Current recipe step:', curr_step, recipe[curr_step])
            print('Input action:', actions[ii])
            print('Is a mistake:', 'No')
            print('*********************')

     else:
            
        if pred[0] == 1:
                executed = True
                print('Current recipe step:', curr_step, recipe[curr_step])
                print('Input action:', actions[ii])
                print('Is a mistake:', 'No')
                print('*********************')
        else:
                print('Current recipe step:', curr_step, recipe[curr_step])
                print('Input action:', actions[ii])
                print('Is a mistake:', 'Yes')
                print('*********************') 


     #print(pred, output, passage)
     input('Next Action')
