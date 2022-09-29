import torch
import string
import nltk
from transformers import pipeline
from os.path import join
from spacy import displacy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
punctuation_marks = string.punctuation


class RecipeTagger:

    def __init__(self, model_path):
        model_path_edwardjross = join(model_path, 'recipe_edwardjross')
        self.model_edwardjross = pipeline('token-classification', model=model_path_edwardjross,
                                          tokenizer=model_path_edwardjross, ignore_labels=[], device=device)
        model_path_flowgraph = join(model_path, 'recipe_flowgraph')
        self.model_flowgraph = pipeline('token-classification', model=model_path_flowgraph,
                                        tokenizer=model_path_flowgraph, ignore_labels=[], device=device)

    def predict_entities(self, text, join_entities=True):
        sentences = nltk.sent_tokenize(text)
        all_tokens = []
        all_tags = []
        for sentence in sentences:
            entities = self.model_edwardjross(sentence)
            tokens_me, tags_me = self._join_sub_entities(entities)
            entities = self.model_flowgraph(sentence)
            tokens_mf, tags_mf = self._join_sub_entities(entities)
            tags_me = self._replace_tags(tags_me)
            tags_mf = self._replace_tags(tags_mf)
            tokens, tags = self._combine_results(tokens_me, tags_me, tokens_mf, tags_mf)
            all_tokens += tokens
            all_tags += tags

        if join_entities:
            all_tokens, all_tags = self._join_entities(all_tokens, all_tags)

        return all_tokens, all_tags

    def _combine_results(self, tokens1, tags1, tokens2, tags2):
        if tokens1 != tokens2:  # Since the tokens are not the same, only return the ones by the best model
            return tokens1, tags1

        for index in range(len(tokens1)):
            tag1 = tags1[index]
            tag2 = tags2[index]

            if tag1 == 'INGREDIENT' and tag2 != 'INGREDIENT':  # If there is no agreement, don't tag it as ingredient
                tags1[index] = 'O'
            if tag2 in {'ACTION', 'TOOL', 'DURATION'}:
                tags1[index] = tag2

        return tokens1, tags1

    def _join_sub_entities(self, entities):
        tokens = []
        tags = []

        for token in entities:
            word = token['word']
            tag = token['entity']
            if word.startswith('▁'):  # It's the first part of the word
                tokens.append(word.lstrip('▁'))
                tags.append(tag)
            elif word in punctuation_marks:
                tokens.append(word)
                tags.append(tag)
            else:
                tokens[-1] = tokens[-1] + word

        return tokens, tags

    def _join_entities(self, tokens, tags):
        new_tokens = []
        new_tags = []
        previous_tag = None

        for token, tag in zip(tokens, tags):
            if tag == previous_tag:
                new_tokens[-1] = new_tokens[-1] + ' ' + token
            else:
                new_tokens.append(token)
                new_tags.append(tag)

            previous_tag = tag

        return new_tokens, new_tags

    def _replace_tags(self, tags):
        new_tags = []
        mappings_edwardjross = {'I-NAME': 'INGREDIENT', 'I-QUANTITY': 'QUANTITY', 'I-UNIT': 'UNIT', 'I-STATE': 'STATE',
                                'I-SIZE': 'SIZE', 'I-TEMP': 'TEMP'}
        mappings_flowgraph = {'I-T': 'TOOL', 'I-Ac': 'ACTION', 'I-Ac2': 'ACTION', 'I-Af': 'ACTION', 'I-At': 'ACTION',
                              'I-D': 'DURATION', 'I-F': 'INGREDIENT', 'I-Q': 'QUANTITY', 'I-Sf': 'STATE'}

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
        colors = ['orange', 'gray', 'pink', 'yellow', 'blue', 'red', 'green', 'magenta', 'cyan']

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
