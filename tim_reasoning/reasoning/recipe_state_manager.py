import sys
import logging
import tim_reasoning.utils as utils
from enum import Enum
from tim_reasoning.reasoning.recipe_tagger import RecipeTagger
from tim_reasoning.reasoning.rule_based_classifier import RuleBasedClassifier
from tim_reasoning.reasoning.bert_classifier import BertClassifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


class StateManager:

    def __init__(self, configs):
        self.recipe_tagger = RecipeTagger(configs['tagger_model_path'])
        self.rule_classifier = RuleBasedClassifier(self.recipe_tagger)
        self.bert_classifier = BertClassifier(configs['bert_classifier_path'])
        self.recipe = None
        self.current_step_index = None
        self.graph_task = None
        self.status = RecipeStatus.NOT_STARTED

    def start_recipe(self, recipe):
        self.recipe = recipe
        self.current_step_index = 0
        self.graph_task = []
        self._build_task_graph()
        self.graph_task[self.current_step_index]['step_status'] = StepStatus.NEW
        self.status = RecipeStatus.IN_PROGRESS
        logger.info('Starting a recipe ...')

        return {
            'step_id': self.current_step_index,
            'step_status': self.graph_task[self.current_step_index]['step_status'].value,
            'step_description': self.graph_task[self.current_step_index]['step_description'],
            'error_status': False,
            'error_description': ''
        }

    def check_status(self, detected_actions, scene_descriptions=None):
        if self.status == RecipeStatus.NOT_STARTED:
            raise SystemError('Call the method "start_steps()" to begin the process.')

        current_step = self.graph_task[self.current_step_index]['step_description']
        logger.info(f'Current step: "{current_step}"')

        if self.status == RecipeStatus.COMPLETED:
            return {
                'step_id': self.current_step_index,
                'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                'step_description': self.graph_task[self.current_step_index]['step_description'],
                'error_status': False,
                'error_description': ''
            }

        valid_actions, exist_actions = self._preprocess_inputs(detected_actions)

        if len(valid_actions) == 0 and exist_actions:  # If there are no valid actions, don't make a decision, just wait for new inputs
            logger.info('No valid actions to be processed')
            return

        if not exist_actions:  # Is the user waiting for instructions?
            if self.graph_task[self.current_step_index]['step_status'] == StepStatus.IN_PROGRESS:  # Was the step executed at least once?
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.COMPLETED  # Mark as a done step

                if self.current_step_index == len(self.graph_task) - 1:  # If recipe completed, don't move
                    self.status = RecipeStatus.COMPLETED
                    return

                self.current_step_index += 1  # Move to the next step
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.NEW

                return {  # Return next step
                    'step_id': self.current_step_index,
                    'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                    'step_description': self.graph_task[self.current_step_index]['step_description'],
                    'error_status': False,
                    'error_description': ''
                }
            else:
                return {  # Return the same step
                    'step_id': self.current_step_index,
                    'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                    'step_description': self.graph_task[self.current_step_index]['step_description'],
                    'error_status': False,
                    'error_description': ''
                }

        mistake, _ = self._has_mistake(current_step, valid_actions)

        if mistake:
            return {
                'step_id': self.current_step_index,
                'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                'step_description': self.graph_task[self.current_step_index]['step_description'],
                'error_status': True,
                'error_description': 'Errors detected in the step'
            }

        else:
            self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
            return {
                'step_id': self.current_step_index,
                'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                'step_description': self.graph_task[self.current_step_index]['step_description'],
                'error_status': False,
                'error_description': ''
            }

    def reset(self):
        self.recipe = None
        self.current_step_index = None
        self.graph_task = None
        self.status = RecipeStatus.NOT_STARTED
        logger.info('Recipe resetted')

    def set_user_feedback(self, new_step_index=None):
        if new_step_index is None:
            new_step_index = self.current_step_index + 1  # Assume it's the next step when new_step_index is None

        for index in range(new_step_index):  # Update previous steps
            self.graph_task[index]['step_status'] = StepStatus.COMPLETED

        self.current_step_index = new_step_index
        self.graph_task[self.current_step_index]['step_status'] = StepStatus.NEW
        logger.info(f'Feedback received, now step index = {self.current_step_index}')

        return {
            'step_id': self.current_step_index,
            'step_status': self.graph_task[self.current_step_index]['step_status'].value,
            'step_description': self.graph_task[self.current_step_index]['step_description'],
            'error_status': False,
            'error_description': ''
        }

    def get_entities(self):
        ingredients_tools = []

        for index, step_data in enumerate(self.graph_task):
            ingredients_tools.append({'step_id': index, 'step_entities': step_data['step_entities']})

        return ingredients_tools

    def _build_task_graph(self, map_entities=True):
        recipe_entity_labels = utils.load_recipe_entity_labels(self.recipe['name'])

        for step in self.recipe['instructions']:
            entities = self._extract_entities(step)
            logger.info(f'Found entities in the step:{str(entities)}')
            if map_entities:
                entities = utils.map_entity_labels(recipe_entity_labels, entities)
                logger.info(f'New names for entities: {str(entities)}')
            self.graph_task.append({'step_description': step, 'step_status': StepStatus.NOT_STARTED,
                                    'step_entities': entities})

    def _has_mistake(self, current_step, detected_actions):
        # Perception will send the top-k actions for a single frame
        bert_score = None

        for detected_action in detected_actions:
            logger.info(f'Evaluating "{detected_action}"...')
            is_mistake_bert, bert_score = self.bert_classifier.is_mistake(current_step, detected_action)
            is_mistake_rule = self.rule_classifier.is_mistake(current_step, detected_action)
            # If there is an agreement of "NO MISTAKE" by both classifier, then it's not a mistake
            # TODO: We are not using an ensemble voting classifier because there are only 2 classifiers, but we should do for n>=3 classifiers
            if not is_mistake_bert and not is_mistake_rule:
                logger.info('Final decision: IT IS NOT A MISTAKE')
                return False, bert_score

        logger.info('Final decision: IT IS A MISTAKE')
        return True, bert_score

    def _preprocess_inputs(self, actions, proba_threshold=0.2):
        valid_actions = []
        exist_actions = False

        for action_description, action_proba in actions:
            if action_proba >= proba_threshold:
                # Split the inputs to have actions in the form: verb + noun
                nouns = action_description.split(', ')
                verb, first_noun = nouns.pop(0).split(' ', 1)
                for noun in [first_noun] + nouns:
                    valid_actions.append(verb + ' ' + noun)

            if action_proba > 0.0:
                exist_actions = True

        logger.info(f'Actions after pre-processing: {str(valid_actions)}')
        return valid_actions, exist_actions

    def _extract_entities(self, step):
        entities = {'ingredients': set(), 'tools': set()}
        tokens, tags = self.recipe_tagger.predict_entities(step)

        for token, tag in zip(tokens, tags):
            if tag == 'INGREDIENT':
                entities['ingredients'].add(token)
            elif tag == 'TOOL':
                entities['tools'].add(token)

        return entities


class RecipeStatus(Enum):
    NOT_STARTED = 'NOT_STARTED'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'


class StepStatus(Enum):
    NOT_STARTED = 'NOT_STARTED'
    NEW = 'NEW'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'
