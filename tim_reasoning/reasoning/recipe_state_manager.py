import sys
import logging
import numpy as np
import tim_reasoning.utils as utils
from enum import Enum
from tim_reasoning.reasoning.recipe_tagger import RecipeTagger
from tim_reasoning.reasoning.rule_based_classifier import RuleBasedClassifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


class StateManager:

    def __init__(self, configs):
        self.recipe_tagger = RecipeTagger(configs['tagger_model_path'])
        self.rule_classifier = RuleBasedClassifier(self.recipe_tagger)
        self.recipe = None
        self.status = RecipeStatus.NOT_STARTED
        self.current_step_index = None
        self.graph_task = None
        self.probability_matrix = None
        self.transition_matrix = None
        self.min_executions = None

    def start_recipe(self, recipe):
        self.recipe = recipe
        self.current_step_index = 0
        self.graph_task = []
        self.probability_matrix = utils.create_matrix(recipe['_id'])
        self.min_executions = self.probability_matrix['step_times']
        self.transition_matrix = np.zeros(self.probability_matrix['matrix'].shape[0])
        self.transition_matrix[0] = 1.0
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

    def check_status(self, detected_actions, detected_objects, detected_steps, use_perception_steps=True):
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
        if use_perception_steps:  # Use perception
            sorted_step_probas = sorted(detected_steps.items(), key=lambda x: x[1], reverse=True)
            detected_step_index = sorted_step_probas[0][0]
            if detected_step_index >= 0:  # negative index is OTHER
                if detected_step_index >= self.current_step_index:
                    self.graph_task[self.current_step_index]['step_status'] = StepStatus.COMPLETED
                self.current_step_index = detected_step_index
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        else:
            self.identify_status(detected_actions)

        return {
            'step_id': self.current_step_index,
            'step_status': self.graph_task[self.current_step_index]['step_status'].value,
            'step_description': self.graph_task[self.current_step_index]['step_description'],
            'error_status': False,
            'error_description': ''
        }

    def identify_status(self, detected_actions, window_size=1, threshold_confidence=0.3):
        self.graph_task[self.current_step_index]['executions'] += 1

        probability_matrix = self.probability_matrix['matrix']
        indexes = self.probability_matrix['indexes']
        vector = np.zeros(len(indexes))

        for action_name, action_proba in detected_actions:
            if action_name in indexes:
                if action_proba >= threshold_confidence:
                    vector[indexes[action_name]] = action_proba

        dot_product = np.dot(probability_matrix, vector)
        dot_product = np.multiply(dot_product, self.transition_matrix).round(5)
        move = self._calculate_move(self.current_step_index, dot_product, window_size)

        if move >= 1:
            if self.graph_task[self.current_step_index]['executions'] > self.min_executions[self.current_step_index]:
                prev = self.current_step_index
                self.transition_matrix[prev] = 0.10
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.COMPLETED
                if self.current_step_index + move < len(self.graph_task):
                    self.graph_task[self.current_step_index + move]['step_status'] = StepStatus.NEW

            else:
                move = 0

        elif move == 0:
            self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS

        elif move <= -1:
            move = 0  # Avoid go back

        next_step = self.current_step_index + 1
        if len(self.graph_task) > next_step:
            self.transition_matrix[next_step] = 1.0

        if 0 <= self.current_step_index + move < len(self.graph_task):
            self.current_step_index += move

    def reset(self):
        self.recipe = None
        self.current_step_index = None
        self.graph_task = None
        self.probability_matrix = None
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

    def _calculate_move(self, current_index, values, window_size):
        windows = [values[current_index]] * (window_size * 2 + 1)

        for i in range(window_size):
            previous_index = current_index - (i + 1)
            next_index = current_index + (i + 1)
            previous_value = values[previous_index] if previous_index >= 0 else -float('inf')
            next_value = values[next_index] if next_index < len(values) else -float('inf')
            windows[window_size - (i + 1)] = previous_value
            windows[window_size + (i + 1)] = next_value

        max_index = np.argmax(windows)
        move = max_index - window_size

        return move

    def _build_task_graph(self, map_entities=True):
        recipe_entity_labels = utils.load_recipe_entity_labels(self.recipe['_id'])

        for step in self.recipe['instructions']:
            entities = self._extract_entities(step)
            logger.info(f'Found entities in the step:{str(entities)}')
            if map_entities:
                entities = utils.map_entity_labels(recipe_entity_labels, entities)
                logger.info(f'New names for entities: {str(entities)}')
            self.graph_task.append({'step_description': step, 'step_status': StepStatus.NOT_STARTED,
                                    'step_entities': entities, 'executions': 0})

    def _detect_error_in_actions(self, detected_actions):
        # Perception will send the top-k actions for a single frame
        current_step = self.graph_task[self.current_step_index]['step_description']

        for detected_action in detected_actions:
            logger.info(f'Evaluating "{detected_action}"...')
            has_error_rule = self.rule_classifier.is_mistake(current_step, detected_action)
            if not has_error_rule:
                logger.info('Final decision: IT IS NOT A ERROR')
                return False

        logger.info('Final decision: IT IS A ERROR')
        return True

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
