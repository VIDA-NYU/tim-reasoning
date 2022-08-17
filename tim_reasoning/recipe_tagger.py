import re
import torch
import string
from transformers import AutoTokenizer, AutoModelForTokenClassification
from os.path import join
from datasets import ClassLabel
from itertools import groupby
from spacy import displacy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
punctuation_marks = string.punctuation


class HuggingFaceModel:

    def __init__(self, model_path, label_names):
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.__model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.__labels = ClassLabel(num_classes=len(label_names), names=label_names)

    def predict(self, sentence):
        inputs = self.__tokenizer(sentence, return_tensors='pt')
        tokens = [self.__tokenizer.convert_ids_to_tokens(x) for x in inputs['input_ids']]
        preds = [self.__labels.int2str(x) for x in self.__model(**inputs.to(device)).logits.argmax(axis=-1)]
        tokens, tags = self._align_token_preds(inputs, tokens, preds)[0]

        return tokens, tags

    def _align_token_preds(self, inputs, tokens, preds):
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


class RecipeTagger:

    def __init__(self, model_path):
        labels_edwardjross = ['O', 'I-QUANTITY', 'I-UNIT', 'I-STATE', 'I-NAME', 'I-DF', 'I-SIZE', 'I-TEMP']
        labels_flowgraph = ['I-Ac', 'I-Ac2', 'I-Af', 'I-At', 'I-D', 'I-F', 'O', 'I-Q', 'I-Sf', 'I-St', 'I-T']
        self.model_edwardjross = HuggingFaceModel(join(model_path, 'recipe_edwardjross'), labels_edwardjross)
        self.model_flowgraph = HuggingFaceModel(join(model_path, 'recipe_flowgraph'), labels_flowgraph)

    def predict_entities(self, text, replace_original_tags=True):
        tokens, tags = self.model_edwardjross.predict(text)
        tokens2, tags2 = self.model_flowgraph.predict(text)

        if replace_original_tags:
            tags = self._replace_tags(tags)
            tags2 = self._replace_tags(tags2)

        new_tags = []
        for token, tag in zip(tokens, tags):
            for to, ta in zip(tokens2, tags2):
                #print(to, ta)
                if token == to and tag == 'O' and ta in {'ACTION', 'TOOL'}:
                    tag = ta

            new_tags.append(tag)

        return tokens, new_tags

    def _replace_tags(self, tags):
        new_tags = []
        mappings_edwardjross = {'I-NAME': 'INGREDIENT', 'I-QUANTITY': 'QUANTITY', 'I-UNIT': 'UNIT', 'I-STATE': 'STATE', 'I-SIZE': 'SIZE', 'I-TEMP': 'TEMP'}
        mappings_flowgraph = {'I-Ac': 'ACTION', 'I-Ac2': 'ACTION', 'I-Af': 'ACTION', 'I-At': 'ACTION', 'I-D': 'DURATION', 'I-F': 'INGREDIENT', 'I-Q': 'QUANTITY', 'I-Sf': 'STATE',  'I-T': 'TOOL'}

        for tag in tags:
            if tag in {'I-DF', 'I-St'}:  # Skip these tags
                tag = 'O'
            elif tag in mappings_edwardjross:
                tag = mappings_edwardjross[tag]
            elif tag in mappings_flowgraph:
                tag = mappings_flowgraph[tag]
            new_tags.append(tag)

        return new_tags

    def plot_entities(self, tokens, tags, display_entities=None):
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
            index += len(token)+1  # Plus 1 because of the empty space

        text_info = [{'text': text.rstrip(), 'ents': entities}]

        entity_names = ['INGREDIENT', 'QUANTITY', 'UNIT', 'STATE', 'SIZE', 'TEMP', 'ACTION', 'TOOL', 'DURATION']
        colors = ['orange', 'gray', 'pink', 'yellow', 'blue', 'red', 'green', 'yellow', 'red']

        if display_entities is None:
            display_entities = entity_names  # Show all the entities

        options = {'ents': [e for e in entity_names if e in display_entities], 'colors': zip(entity_names, colors)}
        displacy.render(text_info, style='ent', manual=True, options=options)

    def extract_action_relations(self, tokens, tags):
        action_relations = []
        action = None
        action_used = False

        for token, tag in zip(tokens, tags):
            if tag == 'ACTION':
                if not action_used and action is not None:
                    action_relations.append((action, None))
                action = token
                action_used = False
            elif tag in {'INGREDIENT', 'TOOL'}:
                object = token.rstrip(punctuation_marks)
                if action is not None:
                    action_relations.append((action, object))
                    action_used = True

        if not action_used and action is not None:
            action_relations.append((action, None))

        return action_relations
