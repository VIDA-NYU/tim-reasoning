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

    def check_status(self, detected_actions, detected_objects):
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

        error_act_status, _ = self._detect_error_in_actions(valid_actions)
        error_obj_status, error_obj_message, error_obj_entities = self._detect_error_in_objects(detected_objects)

        if error_act_status:
            return {
                'step_id': self.current_step_index,
                'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                'step_description': self.graph_task[self.current_step_index]['step_description'],
                'error_status': True,
                'error_description': error_obj_message
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
        recipe_entity_labels = utils.load_recipe_entity_labels(self.recipe['_id'])

        for step in self.recipe['instructions']:
            entities = self._extract_entities(step)
            logger.info(f'Found entities in the step:{str(entities)}')
            if map_entities:
                entities = utils.map_entity_labels(recipe_entity_labels, entities)
                logger.info(f'New names for entities: {str(entities)}')
            self.graph_task.append({'step_description': step, 'step_status': StepStatus.NOT_STARTED,
                                    'step_entities': entities})

    def _detect_error_in_actions(self, detected_actions):
        # Perception will send the top-k actions for a single frame
        current_step = self.graph_task[self.current_step_index]['step_description']
        bert_score = None

        for detected_action in detected_actions:
            logger.info(f'Evaluating "{detected_action}"...')
            has_error_bert, bert_score = self.bert_classifier.is_mistake(current_step, detected_action)
            has_error_rule = self.rule_classifier.is_mistake(current_step, detected_action)
            # If there is an agreement of "NO ERROR" by both classifier, then it's not a error
            # TODO: We are not using an ensemble voting classifier because there are only 2 classifiers, but we should do for n>=3 classifiers
            if not has_error_bert and not has_error_rule:
                logger.info('Final decision: IT IS NOT A ERROR')
                return False, bert_score

        logger.info('Final decision: IT IS A ERROR')
        return True, bert_score

    def _detect_error_in_objects(self, detected_objects):
        tools_in_step = set(self.graph_task[self.current_step_index]['step_entities']['tools'])
        ingredients_in_step = set(self.graph_task[self.current_step_index]['step_entities']['ingredients'])
        error_message = ''
        error_entities = {'ingredients': {'right': [], 'wrong': []}, 'tools': {'right': [], 'wrong': []}}
        has_error = False

        for object_data in detected_objects:
            object_label = object_data['label']

            if object_label in tools_in_step:
                tools_in_step.remove(object_label)

            if object_label in ingredients_in_step:
                ingredients_in_step.remove(object_label)

        if len(ingredients_in_step) > 0:
            error_message = f'You are not using the ingredient: {", ".join(ingredients_in_step)}. '
            has_error = True
            error_entities['ingredients']['right'] = list(ingredients_in_step)

        if len(tools_in_step) > 0:
            error_message += f'You are not using the tool: {", ".join(tools_in_step)}. '
            has_error = True
            error_entities['tools']['right'] = list(tools_in_step)

        return has_error, error_message, error_entities

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
