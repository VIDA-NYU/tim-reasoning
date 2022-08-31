import sys
import logging
from enum import Enum
from tim_reasoning.reasoning.rule_based_classifier import RuleBasedClassifier
from tim_reasoning.reasoning.bert_classifier import BertClassifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self, recipe, configs):
        self.recipe = recipe
        self.rule_classifier = RuleBasedClassifier(configs['rule_classifier_path'])
        self.bert_classifier = BertClassifier(configs['bert_classifier_path'])
        self.graph_task = []  # Initially we are using a simple list for the graph task
        self.status = RecipeStatus.NOT_STARTED
        self.current_step_index = None
        self._build_task_graph()

    def start_steps(self):
        current_step = self.graph_task[0]['step']
        self.status = RecipeStatus.IN_PROGRESS
        self.current_step_index = 0

        return {
            'step_id': self.current_step_index,
            'step_status': StepStatus.IN_PROGRESS,
            'step_description': current_step,
            'error_status': False,
            'error_description': ''
        }

    def _build_task_graph(self):
        for step in self.recipe['steps']:
            self.graph_task.append({'step': step, 'is_completed': False, 'sub_steps_counter': 0})  # TODO: sub_steps_counter is just a placeholder to know when a step is completed

    def check_status(self, detected_actions, scene_descriptions=None):
        if self.status == RecipeStatus.NOT_STARTED:
            raise SystemError('Call the method "start_steps()" to begin the process.')

        if self.status == RecipeStatus.COMPLETED:
            raise SystemError('The recipe has been completed.')

        current_step = self.graph_task[self.current_step_index]['step']
        mistake = self._has_mistake(current_step, detected_actions)

        if mistake:
            return {
                'step_id': self.current_step_index,
                'step_status': StepStatus.IN_PROGRESS,
                'step_description': current_step,
                'error_status': True,
                'error_description': 'Errors detected in the step'
            }

        else:
            self.graph_task[self.current_step_index]['sub_steps_counter'] += 1
            if self.graph_task[self.current_step_index]['sub_steps_counter'] == 2:
                self.graph_task[self.current_step_index]['is_completed'] = True
            # TODO: The 3-lines above is just a simulation to see if the step is completed. We have to improve that

            if self.graph_task[self.current_step_index]['is_completed']:  # Is the step completed?
                self.current_step_index += 1
                if self.current_step_index == len(self.graph_task):  # Is the recipe completed?
                    self.status = RecipeStatus.COMPLETED
                    return {'step': 'Enjoy, you completed the recipe', 'error': False, 'status_message': 'Recipe completed'}
                else:
                    current_step = self.graph_task[self.current_step_index]['step']
                    return {
                        'step_id': self.current_step_index,
                        'step_status': StepStatus.NEW,
                        'step_description': current_step,
                        'error_status': False,
                        'error_description': ''
                    }
            else:
                return {
                    'step_id': self.current_step_index,
                    'step_status': StepStatus.IN_PROGRESS,
                    'step_description': current_step,
                    'error_status': False,
                    'error_description': ''
                }

    def _has_mistake(self, current_step, detected_actions):
        for detected_action in detected_actions:
            is_mistake_bert = self.bert_classifier.is_mistake(current_step, detected_action)
            is_mistake_rule = self.rule_classifier.is_mistake(current_step, detected_action)

            if not is_mistake_bert and not is_mistake_rule:
                return False

        return True


class RecipeStatus(Enum):
    NOT_STARTED = 'NOT_STARTED'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'


class StepStatus(Enum):
    IN_PROGRESS = 'IN_PROGRESS'
    NEW = 'NEW'
