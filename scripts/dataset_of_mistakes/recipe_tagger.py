from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import ClassLabel
from itertools import groupby
from spacy import displacy
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RecipeTagger:
    def __init__(self, model_path):
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.__model = AutoModelForTokenClassification.from_pretrained(model_path)
        label_names = ['O', 'I-QUANTITY', 'I-UNIT', 'I-STATE', 'I-NAME', 'I-DF', 'I-SIZE', 'I-TEMP']
        self.__labels = ClassLabel(num_classes=len(label_names), names=label_names)

    def predict_entities(self, text, replace_original_tags=True):
        inputs = self.__tokenizer(text, return_tensors='pt')
        tokens = [self.__tokenizer.convert_ids_to_tokens(x) for x in inputs['input_ids']]
        preds = [self.__labels.int2str(x) for x in self.__model(**inputs.to(device)).logits.argmax(axis=-1)]
        tokens, tags = self.align_token_preds(inputs, tokens, preds)[0]

        if replace_original_tags:
            new_tags = []
            for tag in tags:
                if tag in {'I-DF'}:  # Skip these tags
                    tag = 'O'
                tag = tag.replace('I-NAME', 'INGREDIENT')
                tag = tag.replace('I-QUANTITY', 'QUANTITY')
                tag = tag.replace('I-UNIT', 'UNIT')
                tag = tag.replace('I-STATE', 'STATE')
                tag = tag.replace('I-SIZE', 'SIZE')
                tag = tag.replace('I-TEMP', 'TEMP')
                new_tags.append(tag)
            tags = new_tags

        return tokens, tags

    def align_token_preds(self, inputs, tokens, preds):
        results = []
        for idx in range(inputs['input_ids'].shape[0]):
            result = []
            for key, group in groupby(zip(inputs.word_ids(idx), tokens[idx], preds[idx]), key=lambda x: x[0]):
                if key is not None:
                    group = list(group)
                    token = ''.join([x[1] for x in group]).lstrip('▁')
                    tag = group[0][2]
                    result.append((token, tag))
            results.append(list(zip(*result)))
        return results

    def plot_entities(self, tokens, tags):
        entities = []
        text = ''
        index = 0

        for i in range(len(tokens)):
            tag = tags[i]
            token = tokens[i]

            if tag != 'O':
                entity_info = {'start': index, 'end': index+len(token), 'label': tag}
                entities.append(entity_info)
            text += token + ' '
            index += len(token)+1  # Plus 1 becuase of the empty space

        text_info = [{'text': text.rstrip(), 'ents': entities}]

        colors = {'INGREDIENT': 'orange', 'QUANTITY': 'gray', 'UNIT': 'pink', 'STATE': 'yellow', 'SIZE': 'blue', 'TEMP': 'green'}
        options = {'ents': ['INGREDIENT', 'QUANTITY', 'UNIT', 'STATE', 'SIZE', 'TEMP'], 'colors':colors}
        displacy.render(text_info, style='ent', manual=True, options=options)
