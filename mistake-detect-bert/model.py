import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from pytorch_transformers import *
from pytorch_transformers.modeling_bert import BertModel, BertEncoder, BertPreTrainedModel
import os


class DeepQDSBertModelHelper(BertPreTrainedModel):
    def __init__(self, encoder_title: BertModel,  args, bert_config):
        super(DeepQDSBertModelHelper, self).__init__(bert_config)
        self.bert_encoder_title = encoder_title
        self.batch_counter = 0
        self.logger = args.logger
        self.args = args
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.title_mlp = nn.Linear(self.config.hidden_size, 2)
        #self.title_mlp.apply(self.init_bert_weights)
        #self.title_mlp_2 = nn.Linear(100, 1)
        #self.title_mlp_2.apply(self.init_bert_weights)


        #self.sigmoid = nn.Sigmoid()

    def log_summary_writer(self, logs: dict, base='Train'):
        if (self.args.no_cuda and self.args.local_rank == -1):
            counter = self.batch_counter
            for key, log in logs.items():
                self.args.summary_writer.add_scalar(
                    f'{base}/{key}', log, counter)
            self.batch_counter = counter + 1

    def forward(self, batch, log=True):
        query_title_ids = batch[0]
        query_title_mask = batch[1]
        query_title_type_ids = batch[2]

        labels = batch[3]
        #ids = batch[4]


        labels = batch[3] if not self.args.fp16 or batch[3] is None else batch[3].half()

        _, title_cls = self.bert_encoder_title(query_title_ids, query_title_type_ids, query_title_mask)
        title_cls = self.dropout(title_cls)
        #title_score =  self.sigmoid(self.title_mlp(title_cls))
        title_score =  self.title_mlp(title_cls)



        if labels is not None:
            criterion = nn.CrossEntropyLoss()

            loss = criterion(
                title_score, labels)
            if log:
                self.log_summary_writer(logs={'loss': loss.item()})
            return loss
        else:
            return title_score



class DeepQDSBertModel:
    def __init__(self, args, config, summary_writer):
        self.config = config
        self.args = args
        self.bert_encoder_title = BertModel.from_pretrained(self.config['bert_model_file'])
        self.bert_config = self.bert_encoder_title.config
        self.network=DeepQDSBertModelHelper(self.bert_encoder_title, self.args, self.bert_config)
        self.device=args.device

    def half(self):
        self.network.half()
    
    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()
    
    def save(self, filename: str):
        network = self.network
        if isinstance(network, nn.DataParallel):
            network = network.module

        return torch.save(self.network.state_dict(), filename)

    def load(self, model_state_dict: str):
        return self.network.load_state_dict(torch.load(model_state_dict, map_location=lambda storage, loc: storage))
        self.network.half()
    