import sys
import torch
import json
import argparse
import logging
from os.path import join, dirname
from tim_reasoning.models.mistake_detect_bert.model import DeepQDSBertModel
from tim_reasoning.models.mistake_detect_bert.dataset import encode_sequence
from pytorch_transformers.tokenization_bert import BertTokenizer


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

CONFIG_PATH = '../models/mistake_detect_bert/configs/config_bert.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BertClassifier:

    def __init__(self, model_path):
        parser = argparse.ArgumentParser('Hotpot Edge Ranking')
        args, unknown = parser.parse_known_args()
        args.fp16 = False

        with open(join(dirname(__file__), CONFIG_PATH), 'r', encoding='utf-8') as fin:
            config = json.load(fin)
            config['bert_model_file'] = join(model_path, 'bert-base-uncased')  # Overwrite to load from disk

        self.bert_tokenizer = BertTokenizer.from_pretrained(config['bert_model_file'])
        self.model = DeepQDSBertModel(args, config, None)
        self.model.network.to(device)
        self.model.load(join(model_path, 'model_finetuned_epoch_0.pt'))
        self.model.eval()

    def is_mistake(self, current_step, detected_action):
        ids, mask, type_ids = encode_sequence(current_step, detected_action, 128, self.bert_tokenizer)
        ids = ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        type_ids = type_ids.unsqueeze(0).to(device)
        batch_eval_data = tuple([ids, mask, type_ids, None, None])
        output = self.model.network(batch_eval_data)
        pred = output.argmax(dim=1).data.cpu().numpy().tolist()
        pred_logit = output.data.cpu().numpy().tolist()[0][1]

        if pred[0] == 0:  # Label '0' means it's a mistake
            logger.info('It is a mistake')
            return True, pred_logit
        else:
            logger.info('It is not a mistake')
            return False, pred_logit
