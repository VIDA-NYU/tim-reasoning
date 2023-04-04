import sys
import logging
import numpy as np
import tim_reasoning.utils as utils
from enum import Enum
from tim_reasoning.reasoning.recipe_tagger import RecipeTagger
from tim_reasoning.reasoning.rule_based_classifier import RuleBasedClassifier
import openai

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
            # If there is an agreement of "NO ERROR" by both classifier, then it's not a error
            # TODO: We are not using an ensemble voting classifier because there are only 2 classifiers, but we should do for n>=3 classifiers
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

class BBNManager:

    def __init__(self, configs):
        #self.recipe_tagger = RecipeTagger(configs['tagger_model_path'])
        #self.rule_classifier = RuleBasedClassifier(self.recipe_tagger)
        #self.bert_classifier = BertClassifier(configs['bert_classifier_path'])
        self.recipe = None
        self.current_step_index = None
        self.graph_task = None
        self.status = RecipeStatus.NOT_STARTED
        self.step_ct = 0
        self.object_track_dict = dict()
        self.step_texts = ['Place tourniquet over affected extremity 2-3 inches above wound site.', 'Pull tourniquet tight.', 'Cinch tourniquet strap.', 'Apply strap to strap body.', 'Turn windless clock wise or counter clockwise until hemorrhage is controlled.', 'Lock windless into the windless keeper.', 'Pull remaining strap over the windless keeper.', 'Secure strap and windless keeper with keeper securing device.', 'Mark time on securing device strap with permanent marker.']
        self.total_objs = [[['tourniquet']], [['tourniquet']], [['tourniquet'], ['strap']], [['strap']], [['windless']], [['windless']], [['strap'], ['windless']], [['strap'], ['windless']], [['pen']] ]
        self.total_actions = [['open tourniquet', 'put tourniquet'], ['pull tourniquet'], ['cinch tourniquet'], ['secure strap'], ['turn windlass'], ['lock windlass'], ['wrap strap'], ['secure strap', 'secure windless'], ['write label']]
        self.finished = []
        self.freeze = False
        self.freeze_count = 0 
        self.is_gpt = False
        

    def error_object_prompt(query: str):
        prompt = ''
        prompt += 'Given an instruction and tools, and also current scene objects, detect whether user makes a mistake: \n'
        prompt += 'Instruction:\n'
        prompt += 'Step 1: Place tourniquet over affected extremity 2-3 inches above wound site.\n'
        prompt += 'Step 2: Pull tourniquet tight.\n'
        prompt += 'Step 3: Apply strap to strap body.\n'
        prompt += 'Step 4: Turn windless clock wise or counter clockwise until hemorrhage is controlled.\n'
        prompt += 'Step 5: Lock windless into the windless keeper.\n'
        prompt += 'Step 6: Pull remaining strap over the windless keeper.\n'
        prompt += 'Step 7: Secure strap and windless keeper with keeper securing device.\n'
        prompt += 'Step 8: Mark time on securing device strap with permanent marker.\n'
        prompt += 'Tools: tourniquet, label, windlass, pen, strap.\n\n'
        prompt += 'Input: Step 1, objects include: tourniquet\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 2, objects include: tourniquet\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 3, objects include: strap\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 3, objects include: pen\n'
        prompt += 'Output: Mistake\n\n'
        prompt += 'Input: Step 4, objects include: windless\n'
        prompt += 'Output: Mistake\n\n'
        prompt += 'Input: Step 4, objects include: knife\n'
        prompt += 'Output: Mistake\n\n'
        prompt += 'Input: Step 5, objects include: windless\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 5, objects include: label\n'
        prompt += 'Output: Mistake\n\n'
        prompt += 'Input: Step 6, objects include: windless\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 6, objects include: strap\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 7, objects include: windlass\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 8, objects include: pen\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 8, objects include: label\n'
        prompt += 'Output: No mistake\n\n'
        prompt += f'Input: {query}\n'
        prompt += 'Output:'
        return prompt


    def error_action_prompt(query: str):
        prompt = ''
        prompt += 'Given an instruction and tools, and also current scene actions, detect whether user makes a mistake: \n'
        prompt += 'Instruction:\n'
        prompt += 'Step 1: Place tourniquet over affected extremity 2-3 inches above wound site.\n'
        prompt += 'Step 2: Pull tourniquet tight.\n'
        prompt += 'Step 3: Apply strap to strap body.\n'
        prompt += 'Step 4: Turn windless clock wise or counter clockwise until hemorrhage is controlled.\n'
        prompt += 'Step 5: Lock windless into the windless keeper.\n'
        prompt += 'Step 6: Pull remaining strap over the windless keeper.\n'
        prompt += 'Step 7: Secure strap and windless keeper with keeper securing device.\n'
        prompt += 'Step 8: Mark time on securing device strap with permanent marker.\n'
        prompt += 'Tools: tourniquet, label, windlass, pen.\n\n'
        prompt += 'Input: Step 1, action: open tourniquet\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 1, action: put tourniquet\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 2, action: pull tourniquet\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 3, action: secure strap\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 3, action: put tourniquet\n'
        prompt += 'Output: Mistake\n\n'
        prompt += 'Input: Step 4, action: turn windlass\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 5, action: lock windlass\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 5, action: secure strap\n'
        prompt += 'Output: Mistake\n\n'
        prompt += 'Input: Step 6, action: wrap strap\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 7, action: secure strap\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 7, action: secure windless\n'
        prompt += 'Output: No mistake\n\n'
        prompt += 'Input: Step 8, action: write label\n'
        prompt += 'Output: No mistake\n\n'
        prompt += f'Input: {query}\n'
        prompt += 'Output:'
        return prompt

        
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
            'error_description': '',
            'error_entities': [], 
            'filtered_objects': []
        }

    ### check actual status
    def check_status(self, detected_actions, detected_objects, detected_steps):
        if self.status == RecipeStatus.NOT_STARTED:
            raise SystemError('Call the method "start_steps()" to begin the process.')

        ### TODO: mapping detected steps to numbers
        new_step_dict = dict()
        for k, v in detected_steps.items():
            if k not in ['start', 'end']:
               new_step_dict[k] = v 



        current_step = self.graph_task[self.current_step_index]['step_description']
        logger.info(f'Current step: "{current_step}"')
        self.step_ct += 1
        if self.freeze:
            self.freeze_count += 1
        if self.freeze_count > 15:
            self.freeze = False
            self.freeze_count = 0
        if not self.freeze:
            final_step = sorted(new_step_dict.items(), key=lambda x: x[1], reverse=True)[0]
            self.current_step_index = self.step_texts.index(final_step)

        

        valid_actions, _ = self._preprocess_inputs(detected_actions)
        logger.info(f'let us run: "{current_step}"')

        valid_objects, exist_objects, _ = self._preprocess_input_objects(detected_objects)

        if len(valid_objects) == 0 and exist_objects:  # If there are no valid actions, don't make a decision, just wait for new inputs
            logger.info('No valid objects to be processed')
            return {  # Return next step
                    'step_id': self.current_step_index,
                    'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                    'step_description': self.graph_task[self.current_step_index]['step_description'],
                    'error_status': False,
                    'error_description': '',
                    'error_entities': [],
                    'filtered_objects': []
                }

        else:
            self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
            if self.is_gpt:
                error_act_status = self._detect_error_in_actions(valid_actions)
                error_obj_status= self._detect_error_in_objects(valid_objects)
            else:
                if valid_actions not in self.total_actions[self.current_step_index]:
                    error_act_status = True
                else:
                    error_act_status = False
                error_obj_status = True
                objects_in_step = self.total_objs[self.current_step_index]
                for objs in objects_in_step:
                    if objs[0] in detected_objects:
                        error_obj_status = False


            if error_obj_status and error_act_status:
                final_error = True
            else:
                final_error = False
            self.last_error = final_error
            return {
                'step_id': self.current_step_index,
                'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                'step_description': self.graph_task[self.current_step_index]['step_description'],
                'error_status': final_error,
                'error_description': '',
                'error_entities': '',
                'filtered_objects': valid_objects
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
        self.freeze = True

        return {
            'step_id': self.current_step_index,
            'step_status': self.graph_task[self.current_step_index]['step_status'].value,
            'step_description': self.graph_task[self.current_step_index]['step_description'],
            'error_status': False,
            'error_description': '',
            'error_entities': [],
            'filtered_objects': []
        }


    def _build_task_graph(self):
        for ii in range(9):
            self.graph_task.append({'step_description': self.step_texts[ii], 'step_status': StepStatus.NOT_STARTED})

    def _detect_error_in_actions(self, detected_actions):
        # Perception will send the top-k actions for a single frame
        input_str_act = 'Step ' + str(self.current_step_index) + ' action: ' + detected_actions[0] 
        try:
            response1_act = openai.Completion.create(
            engine=engine,
            prompt= self.error_action_prompt(input_str_act),
            max_tokens=1000,
            logprobs=1,
            temperature=0.,
            # temperature=0.,
            stream=False,
            stop=["<|endoftext|>", "\n\n"]
            # stop=["<|endoftext|>"]
            )
        except Exception as e:
            print(e)
            exit(0)
        error_out_action = response1_act['choices'][0]["text"].strip()
        if 'No mistake' in error_out_action:
            return False
        return True

        
    def _detect_error_in_objects(self, detected_objects):
        
        
        input_str_obj = 'Step ' + str(self.current_step_index) + ', objects include: ' + ', '.join(list(set(detected_objects))) 
        try:
            response = openai.Completion.create(
            engine=engine,
            prompt=self.error_object_prompt(input_str_obj),
            max_tokens=1000,
            logprobs=1,
            temperature=0.,
            # temperature=0.,
            stream=False,
            stop=["<|endoftext|>", "\n\n"]
            # stop=["<|endoftext|>"]
            )
        except Exception as e:
            print(e)
            exit(0)
        error_out_obj = response['choices'][0]["text"].strip()
        if 'No mistake' in error_out_obj:
            return False
        return True

    def _preprocess_input_objects(self, object_boxes):
        valid_objects = []
        exist_objects = False
        score_list = []
        min_distince = dict()
        has_added_set = set()
        new_all_boxes = []
        
        for box in object_boxes:
            if box['hoi_iou'] > 0:
                exist_objects = True
            for o_idx, obj_pred in enumerate(box['labels']):
                if self.step_ct > 0:
                    box_class_score = box['confidences'][o_idx] ** 2 / box['box_confidences'][o_idx]
                    if obj_pred not in self.object_track_dict:
                        self.object_track_dict[obj_pred] = {'indices':[box['xyxyn']], 'score': [box_class_score]}
                        min_distince[obj_pred] = -1
                        has_added_set.add(obj_pred)
                        score_list.append(np.average(self.object_track_dict[obj_pred]['score']))
                    else:

                        avg_indices = []
                        for iii in range(4):
                            avg_indices.append(np.average([iitem[iii] for iitem in self.object_track_dict[obj_pred]['indices']]))
                        avg_score = np.average(self.object_track_dict[obj_pred]['score'])
                        
                        dist = np.sqrt(np.abs(avg_indices[0] - box['xyxyn'][0]) ** 2 +  np.abs(avg_indices[1] - box['xyxyn'][1])** 2 + np.abs(avg_indices[2] - box['xyxyn'][2]) ** 2+ np.abs(avg_indices[3] - box['xyxyn'][3])** 2)
                        ### TODO: need to tune this distance
                        if dist > 0.1:
                            if box_class_score >  avg_score + 0.2:
                                self.object_track_dict[obj_pred] = {'indices':[box['xyxyn']], 'score': [box_class_score]}
                                has_added_set.add(obj_pred)
                                min_distince[obj_pred] = dist
                            score_list.append(box_class_score)
                        else:
                            if obj_pred not in has_added_set:
                                min_distince[obj_pred] = dist 
                                if len(self.object_track_dict[obj_pred]['score']) >= 30:
                                    self.object_track_dict[obj_pred]['score'].append(box_class_score)
                                    self.object_track_dict[obj_pred]['indices'].append(box['xyxyn'])
                                    self.object_track_dict[obj_pred]['indices'].pop(0)
                                    self.object_track_dict[obj_pred]['score'].pop(0)
                                else:
                                    self.object_track_dict[obj_pred]['score'].append(box_class_score)
                                    self.object_track_dict[obj_pred]['indices'].append(box['xyxyn'])
                            else:
                                if dist < min_distince[obj_pred] - 0.05 and box_class_score > self.object_track_dict[obj_pred]['score'][-1] - 0.1:
                                    min_distince[obj_pred] = dist
                                    self.object_track_dict[obj_pred]['score'][-1]= box_class_score
                                    self.object_track_dict[obj_pred]['indices'][-1] = box['xyxyn']


                            score_list.append(np.average(self.object_track_dict[obj_pred]['score']))
                            has_added_set.add(obj_pred)
                             
                
                elif self.step_ct == 0:
                    score_list.append(box_class_score)

                

                new_all_boxes.append({'indices': box['xyxyn'], 'prediction': obj_pred, 'score':box_class_score, 'box_score':  box['box_confidences'][o_idx], 'hoi_iou': box['hoi_iou']})


        sorted_idx = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)

        final_indices = []
        final_label = []
        final_score = []
        if exist_objects:
            for b_idx in sorted_idx:
                item = new_all_boxes[b_idx]
            #for item in detic_pred[ct]:
                idx = str(item['indices'][0]) + '##' +  str(item['indices'][1]) + '##' + str(item['indices'][2]) + '##' + str(item['indices'][3]) + '##'
                if self.step_ct > 10:
                    if item['box_score'] < 0.3:
                        continue
                    
                    if score_list[b_idx] < 0.05:
                        continue
                if item['hoi_iou'] < 0.1:
                    continue
                if item['prediction'] in final_label:
                    continue
                if idx in final_indices:
                    continue


                final_label.append(item['prediction'])
                final_indices.append(idx)
                valid_objects.append(item['prediction'])
                final_score.append(item['score'])

                if self.step_ct == 0:
                    self.object_track_dict[item['prediction']] = {'indices':[item['indices']], 'score': [score_list[b_idx]]}


        logger.info(f'Objects after pre-processing: {str(valid_objects)}')
        logger.info(f'Objects after pre-processing11: {exist_objects}')
        logger.info(f'Objects after pre-processing11: {str(final_score)}')
        return valid_objects, exist_objects, final_score

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



class RecipeStatus(Enum):
    NOT_STARTED = 'NOT_STARTED'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'


class StepStatus(Enum):
    NOT_STARTED = 'NOT_STARTED'
    NEW = 'NEW'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'
