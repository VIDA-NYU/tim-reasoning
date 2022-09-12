import sys
import logging
from enum import Enum
from tim_reasoning.reasoning.rule_based_classifier import RuleBasedClassifier
from tim_reasoning.reasoning.bert_classifier import BertClassifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self, configs):
        self.rule_classifier = RuleBasedClassifier(configs['rule_classifier_path'])
        self.bert_classifier = BertClassifier(configs['bert_classifier_path'])
        self.recipe = None
        self.current_step_index = None
        self.graph_task = []  # Initially we are using a simple list for the graph task
        self.status = RecipeStatus.NOT_STARTED

    def start_recipe(self, recipe):
        self.recipe = recipe
        self.current_step_index = 0
        self._build_task_graph()
        current_step = self.graph_task[self.current_step_index]['step_description']
        self.status = RecipeStatus.IN_PROGRESS

        return {
            'step_id': self.current_step_index,
            'step_status': StepStatus.IN_PROGRESS.value,
            'step_description': current_step,
            'error_status': False,
            'error_description': ''
        }

    def check_status(self, detected_actions, scene_descriptions=None):
        if self.status == RecipeStatus.NOT_STARTED:
            raise SystemError('Call the method "start_steps()" to begin the process.')

        if self.status == RecipeStatus.COMPLETED:
            current_step = self.graph_task[self.current_step_index]['step_description']
            return {
                'step_id': self.current_step_index,
                'step_status': StepStatus.LAST.value,
                'step_description': current_step,
                'error_status': False,
                'error_description': ''
            }

        current_step = self.graph_task[self.current_step_index]['step_description']
        mistake, curr_step_score = self._has_mistake(current_step, detected_actions)

        if mistake:
            if self.graph_task[self.current_step_index]['is_step_completed']:
                next_step = self.graph_task[self.current_step_index + 1]['step_description']
                next_step_mistake, next_step_score = self._has_mistake(next_step, detected_actions)
                # go to next step if not a mistake comparing with next step instruction. 
                if not next_step_mistake:
                    self.current_step_index += 1

            current_step = self.graph_task[self.current_step_index]['step_description']

            return {
                'step_id': self.current_step_index,
                'step_status': StepStatus.IN_PROGRESS.value,
                'step_description': current_step,
                'error_status': True,
                'error_description': 'Errors detected in the step'
            }

        else:
            self.graph_task[self.current_step_index]['is_step_completed'] = True
            # TODO: This assumes that every step is completed after it's execute once. However, there are some steps
            #  that need more than one execution, e.g. steps that requires 5 minutes to be completed.

            if self.graph_task[self.current_step_index]['is_step_completed']:  # Is the step completed?
                if self.current_step_index == len(self.graph_task) - 1:  # Is the recipe completed?
                    self.status = RecipeStatus.COMPLETED

                    return {
                        'step_id': self.current_step_index,
                        'step_status': StepStatus.LAST.value,
                        'step_description': current_step,
                        'error_status': False,
                        'error_description': ''
                    }

                else:
                    # ask classifier to output mistakes for next step
                    next_step = self.graph_task[self.current_step_index + 1]['step_description']
                    next_step_mistake, next_step_score = self._has_mistake(next_step, detected_actions)
                    # go to next step if next step score is higher than current step score
                    if not next_step_mistake and next_step_score > curr_step_score: 
                        self.current_step_index += 1
                    
                    current_step = self.graph_task[self.current_step_index]['step_description']

                    return {
                        'step_id': self.current_step_index,
                        'step_status': StepStatus.NEW.value,
                        'step_description': current_step,
                        'error_status': False,
                        'error_description': ''
                    }
            else:
                return {
                    'step_id': self.current_step_index,
                    'step_status': StepStatus.IN_PROGRESS.value,
                    'step_description': current_step,
                    'error_status': False,
                    'error_description': ''
                }

    def reset(self):
        self.recipe = None
        self.current_step_index = None
        self.graph_task = []
        self.status = RecipeStatus.NOT_STARTED

    def _build_task_graph(self):
        for step in self.recipe['steps']:
            self.graph_task.append({'step_description': step, 'is_step_completed': False})

    def _has_mistake(self, current_step, detected_actions):
        # Perception will send the top-k actions for a single frame
        bert_score = None

        for detected_action in detected_actions:
            logger.info('Evaluating "%s"...' % detected_action)
            is_mistake_bert, bert_score = self.bert_classifier.is_mistake(current_step, detected_action)
            is_mistake_rule = self.rule_classifier.is_mistake(current_step, detected_action)
            # If there is an agreement of "NO MISTAKE" by both classifier, then it's not a mistake
            # TODO: We are not using an ensemble voting classifier because there are only 2 classifiers, but we should for n>=3 classifiers
            if not is_mistake_bert and not is_mistake_rule:
                logger.info('Final decision: IT IS NOT A MISTAKE')
                return False, bert_score

        logger.info('Final decision: IT IS A MISTAKE')
        return True, bert_score


class RecipeStatus(Enum):
    NOT_STARTED = 'NOT_STARTED'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'


class StepStatus(Enum):
    IN_PROGRESS = 'IN_PROGRESS'
    NEW = 'NEW'
    LAST = 'LAST'
