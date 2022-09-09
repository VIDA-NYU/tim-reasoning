import torch
import argparse
import os
import json
import pprint
from typing import List
from torch.utils.data import DataLoader, Dataset
import argparse
import numpy as np
from collections import defaultdict
from multiprocessing import Queue
import time
import pickle
from model import DeepQDSBertModel
from dataset import HotpotSpanDataset
import logging

import pdb
import random
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser("Hotpot Edge Ranking")
parser.add_argument("--config-file", "--cf",
                    help="pointer to the configuration file of the experiment", type=str, required=True)
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                    "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=6,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--do_lower_case",
                    default=True,
                    action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument('--fp16',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--deepscale',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--test',
                    default=False,
                    action='store_true',
                    help="Whether on test mode")


def sort_batch(batch, lengths):
    sorted_lengths, new_indices = torch.sort(lengths, descending=True)
    batch = batch[new_indices]
    _, original_indices = new_indices.sort(dim=0)
    return batch, sorted_lengths, original_indices




def get_dataloader(dataset, batch_size, num_workers=0):
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      pin_memory=config_system['device'] == 'cuda',
                      num_workers=num_workers)



def evaluation_test(eval_file, isTrain=False):
    dataset = HotpotSpanDataset(eval_file, config_model, isTrain, args.tokenizer)
    dataloader = get_dataloader(dataset, config_training["test_batch_size"])
    correct = 0 
    total = 0
    for batch in tqdm(dataloader):    
        batch[0] = batch[0].to(device)
        
        batch[1] = batch[1].to(device)
        batch[2] = batch[2].to(device)
        batch_labels = batch[3].data.cpu().numpy().tolist()
        batch_eval_data = tuple([batch[0], batch[1], batch[2], None, batch[4]])
        output = model.network(batch_eval_data)
        predicted_lbs = output.argmax(dim=1).data.cpu().numpy().tolist()
        for ii in range(len(predicted_lbs)):
            if predicted_lbs[ii] == batch_labels[ii]:
                correct += 1
            total += 1


    
    accu = correct / total
    logging.info("********* TOP1 accuracy ************{}".format(accu))



    return accu
    
def test(index):
    model.eval()
    eval_file = config_system['validation_data']
    auc = evaluation_test(eval_file)


def train(index, config, best_score):
    model.train()
    global global_step
    dataset = HotpotSpanDataset(config_system['train_data'], config_model, True, args.tokenizer)
    print(len(dataset))
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset=dataset, sampler=train_sampler, batch_size=config_training["train_batch_size"], pin_memory=config_system['device'] == 'cuda',
                      num_workers=8)


    print_loss = 0 

    for step, batch in enumerate(tqdm(dataloader)):
        batch[0] = batch[0].to(device)
        batch[1] = batch[1].to(device)
        batch[2] = batch[2].to(device)
        batch[3] = batch[3].to(device)

        #batch[4] = batch[4].to(device)
        batch = tuple([batch[0], batch[1], batch[2], batch[3]])
        loss = model.network(batch)
        if args.n_gpu > 1:
            loss = loss.mean()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        print_loss +=  loss.data.cpu().numpy()


        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        if (step + 1) % 1000 == 0:
            logging.info("********* loss ************{}".format(print_loss))
            print_loss = 0
        if (step + 1) % 3500 == 0:    
            model.eval()
            eval_file = config_system['test_data']
            auc = evaluation_test(eval_file, isTrain=True)
            if auc > best_score:
                best_score = auc
                model.save(os.path.join(base_dir, config['name'], "saved_models/model_finetuned_epoch_{}.pt".format(0)))

        model.train()
    return best_score

  

args = parser.parse_args()
config = json.load(open(args.config_file, 'r', encoding="utf-8"))
base_dir = config['system']['base_dir']
os.makedirs(os.path.join(base_dir, config['name']), exist_ok=True)
os.makedirs(os.path.join(base_dir, config['name'], "saved_models/"), exist_ok=True)
config_training = config["training"]
config_system = config["system"]
config_model = config["model"]

logging.info("********* Model Configuration ************")
pprint.pprint(config)
if config_model.get("bert-model", False):
    # Prepare Logger
    #logger = Logger(cuda=torch.cuda.is_available() and not args.no_cuda)
    args.logger = logger
    args.config = config

    #print("Running Config File: ", config['name'])

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info(
                "16-bits distributed training not officially supported but seems to be working.")
            args.fp16 = True  # (see https://github.com/pytorch/pytorch/pull/13496)
    print(n_gpu)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        config["training"]["train_batch_size"] / args.gradient_accumulation_steps)
    args.max_seq_length = config_model["bert_max_len"]

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #logger.info
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare Summary Writer and saved_models path
    #if True:
    #    summary_writer = get_sample_writer(
    #        name=config['name'], base=base_dir)
    #    args.summary_writer = summary_writer

    # set device
    args.device = device
    args.n_gpu = n_gpu

    # Loading Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
    args.tokenizer = tokenizer
    if args.test:
        model = DeepQDSBertModel(args, config, None)
        model.network.to(device)
        if n_gpu > 1:
            model.network = nn.DataParallel(model.network)
        model.load(os.path.join(base_dir, config['name'], "saved_models/model_finetuned_epoch_{}.pt".format(0)))
        model.eval()
        eval_file = config_system['test_data']
        auc = evaluation_test(eval_file)
        exit()



    # Loading Model
    model = DeepQDSBertModel(args, config, None)

    if args.fp16:
        model.half()
    model.network.to(device)

    if args.local_rank != -1:
        try:
                logger.info(
                    "***** Using Default Apex Distributed Data Parallel *****")
                from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    elif n_gpu > 1:
        model.network = nn.DataParallel(model.network)

    # Prepare Optimizer
    param_optimizer = list(model.network.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config["training"]["learning_rate"])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config["training"]["warmup_proportion"], t_total=config["training"]["total_training_steps"])


    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer, FusedAdam
        except:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=config["training"]["learning_rate"],
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=args.loss_scale)

    global_step = 0

best_score = -1
for index in range(config['training']['epochs']):
    best_score = train(index, config, best_score)
