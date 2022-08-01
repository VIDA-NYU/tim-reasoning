import re
import torch
import string
from transformers import AutoTokenizer, AutoModelForTokenClassification
from allennlp.predictors.predictor import Predictor
from datasets import ClassLabel
from itertools import groupby
from spacy import displacy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
punctuation_marks = string.punctuation


class RecipeTagger:
    def __init__(self, ner_model_path, srl_model_path):
        self.__tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
        self.__model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
        label_names = ['O', 'I-QUANTITY', 'I-UNIT', 'I-STATE', 'I-NAME', 'I-DF', 'I-SIZE', 'I-TEMP']
        self.__labels = ClassLabel(num_classes=len(label_names), names=label_names)
        self.__predictor = Predictor.from_path(srl_model_path)

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

        # Use the other model to identify actions
        # TODO: Fine tune the RoBERTa model to identify actions and utensils
        action_phrases = self.extract_action_phrases(' '.join(tokens))
        for action, phrase in action_phrases:
            for index in range(len(tokens)):
                if action == tokens[index] and tags[index] == 'O':
                    tags[index] = 'ACTION'

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

        entity_names = ['INGREDIENT', 'QUANTITY', 'UNIT', 'STATE', 'SIZE', 'TEMP', 'ACTION']
        colors = ['orange', 'gray', 'pink', 'yellow', 'blue', 'red', 'green']

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
            elif tag in {'INGREDIENT', 'UTENSIL'}:
                object = token.rstrip(punctuation_marks)
                if action is not None:
                    action_relations.append((action, object))
                    action_used = True

        if not action_used and action is not None:
            action_relations.append((action, None))

        return action_relations

    def extract_action_phrases(self, sentence):
        annotations = self.__predictor.predict(sentence=sentence)
        if len(annotations['verbs']) == 0:
            annotations = self.__predictor.predict(sentence=self.convert_to_verb(sentence))

        # print(annotations)
        action_phrases = []

        for verb_info in annotations['verbs']:
            description = verb_info['description']
            roles = re.findall('\[[^\]]+\]', description)  # Find all ARGs
            action = verb_info['verb']

            for role in roles:
                match = re.match(r'^\[ARG\d+: (.+)\]$', role)
                if match:
                    objects_in_action = match.group(1)
                    action_phrases.append((action, objects_in_action))

        return action_phrases

    def convert_to_verb(self, sentence):
        new_sentence = 'To ' + sentence
        return new_sentence
