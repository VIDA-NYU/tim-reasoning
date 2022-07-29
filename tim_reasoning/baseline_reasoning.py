from allennlp.predictors.predictor import Predictor
import spacy
import nltk
import logging
import re

# Change level of logging for all modules
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

nlp = spacy.load('en_core_web_lg')


class BaselineReasoning:

    def __init__(self, recipe, model_path):
        self.recipe = recipe
        self.graph_task = []
        self.current_step = 0
        self.predictor = PredictorSingleton(model_path).get_predictor()

    def extract_actions_objects(self):
        sentences = nltk.sent_tokenize(self.recipe)
        actions = []
        objects = []
        step = 1
        print('{0:15} {1:15} {2}'.format('STEP', 'ACTION', 'OBJECTS'))
        for sentence in sentences:
            annotations = self.predictor.predict(sentence=sentence)
            for verb_info in annotations['verbs']:
                description = verb_info['description']
                roles = re.findall('\[[^\]]+\]', description)  # Find all ARGs
                action = verb_info['verb'].lower()
                objects_in_action = []

                for role in roles:
                    match = re.match(r'^\[ARG\d: (.+)\]$', role)
                    if match:
                        object_in_action = self.clean_text(match.group(1))
                        if object_in_action:
                            objects_in_action.append(object_in_action)
                if len(objects_in_action) > 0:
                    actions.append(action)
                    objects.append(objects_in_action)
                    print('Step {0:10} {1:15} {2}'.format(str(step), action, ', '.join(objects_in_action)))
                    step += 1

        return actions, objects

    def build_task_graph(self):
        actions, objects = self.extract_actions_objects()
        for i in range(len(actions)):
            step = {'action': actions[i], 'objects': objects[i],
                    'instruction': '%s %s' % (actions[i].capitalize(), objects[i][0]),
                    'description': '%s %s' % (actions[i], ' '.join(objects[i])), 'is_done': False}
            self.graph_task.append(step)

        print('Graph task built (%d steps)' % len(self.graph_task))

    def begin_instructions(self, step=0):
        instruction = self.graph_task[step]['instruction']

        return instruction

    def identify_step(self, objects, action='', threshold=0.7):
        use_action = action != ''
        description = '%s %s' % (action, ' '.join(objects))
        text_query = nlp(description)
        max_similarity = float('-inf')
        current_step = None

        for index, step in enumerate(self.graph_task):
            if step['is_done']:  # Ignore steps already done
                continue
            if use_action:
                text_step = nlp(step['description'])
            else:
                text_step = nlp(' '.join(step['objects']))
            similarity = text_query.similarity(text_step)
            #print(text_step, '%s' %step['instruction'], similarity)

            if similarity > max_similarity and similarity >= threshold:
                max_similarity = similarity
                current_step = index

        return current_step

    def next_instruction(self, objects, action=''):
        identified_step = self.identify_step(objects, action)
        instruction = None

        if identified_step is not None:
            if identified_step == self.current_step:
                instruction = 'Next step is: "%s"' % self.graph_task[self.current_step + 1]['instruction']
                self.graph_task[self.current_step]['is_done'] = True
                self.current_step += 1
                if self.current_step + 1 == len(self.graph_task):
                    instruction += ', and that is all!'
            else:
                instruction = 'You are in step "%s" (Step %d).\n' % \
                              (self.graph_task[identified_step]['instruction'], identified_step + 1) + \
                              'However, you should be in step "%s" (Step %d)' % \
                              (self.graph_task[self.current_step]['instruction'], self.current_step + 1)
        else:
            instruction = 'Step not recognized, please provide other objects and/or actions. You should be in step ' \
                          '"%s" (Step %d)' % (self.graph_task[self.current_step]['instruction'], self.current_step + 1)

        return instruction

    def clean_text(self, text):
        docx = nlp(text)
        nouns = []

        for token in docx:
            if not token.is_stop and not token.is_punct and token.pos_ == 'NOUN':
                nouns.append(token.text)

        if len(nouns) == 0:
            return None

        return ' '.join(nouns)


class PredictorSingleton:
    _singleton = None
    predictor = None

    def __new__(cls, model_path):
        if not cls._singleton:
            cls._singleton = super(PredictorSingleton, cls).__new__(cls)
            cls.predictor = Predictor.from_path(model_path)

        return cls._singleton

    def get_predictor(self):
        return self.predictor
