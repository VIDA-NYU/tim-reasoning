import sys
import logging
import tim_reasoning.utils as utils
from enum import Enum
from tim_reasoning.reasoning.recipe_tagger import RecipeTagger
from tim_reasoning.reasoning.rule_based_classifier import RuleBasedClassifier
import numpy as np
from collections import Counter
#from tim_reasoning.reasoning.bert_classifier import BertClassifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


class StateManager:

    def __init__(self, configs):
        self.recipe_tagger = RecipeTagger(configs['tagger_model_path'])
        self.rule_classifier = RuleBasedClassifier(self.recipe_tagger)
        #self.bert_classifier = BertClassifier(configs['bert_classifier_path'])
        self.recipe = None
        self.current_step_index = None
        self.graph_task = None
        self.status = RecipeStatus.NOT_STARTED
        self.step_ct = 0
        self.object_track_dict = dict()
        self.no_act_counter = Counter()
        self.act_counter = Counter()
        self.main_step_exe = []
        self.main_steps = []
        

    def start_recipe(self, recipe):
        self.recipe = recipe
        self.current_step_index = 0
        self.graph_task = []
        self._build_task_graph()
        self.graph_task[self.current_step_index]['step_status'] = StepStatus.NEW
        for ii in range(len(self.recipe['step_objects'])):
            exe_st = [0] * len(self.recipe['step_objects'][ii])
            self.main_step_exe.append(exe_st)
        

        self.status = RecipeStatus.IN_PROGRESS
        logger.info('Starting a recipe ...')

        return {
            'step_id': self.current_step_index,
            'step_status': self.graph_task[self.current_step_index]['step_status'].value,
            'step_description': self.graph_task[self.current_step_index]['step_description'],
            'error_status': False,
            'error_description': '',
            'error_entities': []
        }

    ### check actual status
    def check_status(self, detected_actions, detected_objects):
        if self.status == RecipeStatus.NOT_STARTED:
            raise SystemError('Call the method "start_steps()" to begin the process.')

        current_step = self.graph_task[self.current_step_index]['step_description']
        logger.info(f'Current step: "{current_step}"')
        self.step_ct += 1
        
        ### CHEN: I think this is dangerous 
    
        if self.status == RecipeStatus.COMPLETED:
            return {
                'step_id': self.current_step_index,
                'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                'step_description': self.graph_task[self.current_step_index]['step_description'],
                'error_status': False,
                'error_description': '',
                'error_entities': []
            }
        

        #valid_actions, exist_actions = self._preprocess_inputs(detected_actions)

        valid_objects, exist_objects, object_scores = self._preprocess_input_objects(detected_objects)

        ## actually this should be an error
        if len(valid_objects) and exist_objects:  # If there are no valid actions, don't make a decision, just wait for new inputs
            logger.info('No valid actions to be processed')
            return
        


        if not exist_objects:  # Is the user waiting for instructions?
            self.no_act_counter[self.current_step_index] += 1
            if self.graph_task[self.current_step_index]['step_status'] == StepStatus.COMPLETED and self.current_step_index < len(self.graph_task) - 1 and self.act_counter[self.current_step_index] >= 50 and self.no_act_counter[self.current_step_index] >= 20:
                if self.current_step_index == len(self.graph_task) - 1:  # If recipe completed, don't move
                    self.status = RecipeStatus.COMPLETED
                    return
                self.current_step_index += 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.NEW

                return {  # Return next step
                    'step_id': self.current_step_index,
                    'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                    'step_description': self.graph_task[self.current_step_index]['step_description'],
                    'error_status': False,
                    'error_description': '',
                    'error_entities': []
                }
            else:
                return {  # Return the same step
                    'step_id': self.current_step_index,
                    'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                    'step_description': self.graph_task[self.current_step_index]['step_description'],
                    'error_status': False,
                    'error_description': '',
                    'error_entities': []
                }
        else:
            self.no_act_counter[self.current_step_index] = 0
            self.act_counter[self.current_step_index] += 1


            #error_act_status, _ = self._detect_error_in_actions(valid_actions)
            error_obj_status, error_obj_message, error_obj_entities = self._detect_error_in_objects(detected_objects, object_scores)

            if error_obj_status:
                return {
                    'step_id': self.current_step_index,
                    'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                    'step_description': self.graph_task[self.current_step_index]['step_description'],
                    'error_status': True,
                    'error_description': error_obj_message,
                    'error_entities': error_obj_entities
                }

            else:

                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
                all_finished = True
                for jj in self.main_steps[self.current_step_index]:
                    if self.main_step_exe[self.current_step_index][jj] == 0:
                        all_finished = False
                if all_finished:
                    self.graph_task[self.current_step_index]['step_status'] = StepStatus.COMPLETED
                return {
                    'step_id': self.current_step_index,
                    'step_status': self.graph_task[self.current_step_index]['step_status'].value,
                    'step_description': self.graph_task[self.current_step_index]['step_description'],
                    'error_status': False,
                    'error_description': '',
                    'error_entities': []
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
            'error_description': '',
            'error_entities': []
        }

    def get_entities(self):
        ingredients_tools = []

        for index, step_data in enumerate(self.graph_task):
            ingredients_tools.append({'step_id': index, 'step_entities': step_data['step_entities']})

        return ingredients_tools

    def _build_task_graph(self, map_entities=True):
        recipe_entity_labels = utils.load_recipe_entity_labels(self.recipe['_id'])
        self.main_steps = self.recipe['main_steps']
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
            #has_error_bert, bert_score = self.bert_classifier.is_mistake(current_step, detected_action)
            has_error_rule = self.rule_classifier.is_mistake(current_step, detected_action)
            # If there is an agreement of "NO ERROR" by both classifier, then it's not a error
            # TODO: We are not using an ensemble voting classifier because there are only 2 classifiers, but we should do for n>=3 classifiers
            if not has_error_rule:
                logger.info('Final decision: IT IS NOT A ERROR')
                return False, bert_score

        logger.info('Final decision: IT IS A ERROR')
        return True, bert_score

    def _detect_error_in_objects(self, detected_objects, object_scores):
        objects_in_step = self.recipe['step_objects'][self.current_step_index]
        #ingredients_in_step = self.graph_task[self.current_step_index]['step_entities']['ingredients']
        object_in_step_set = set()
        error_message = ''
        error_entities = {'ingredients': {'right': [], 'wrong': []}, 'tools': {'right': [], 'wrong': []}}
        has_error = True
        for jj in range(len(objects_in_step)):
            ele_list = objects_in_step[jj]
            h_error = 0
            for ele in ele_list:
                object_in_step_set.add(ele)
                if ele not in detected_objects:
                    h_error = 1
            
            if h_error == 0:
                has_error = False
                self.main_step_exe[self.current_step_index][jj] = 1
        

        if has_error and self.status == RecipeStatus.NEW and self.current_step_index > 0:
            objects_last_step = self.recipe['step_objects'][self.current_step_index - 1]
            for jj in range(len(objects_last_step)):
                ele_list = objects_last_step[jj]
                h_error = 0
                for ele in ele_list:
                    if ele not in detected_objects:
                        h_error = 1
                
                if h_error == 0:
                    has_error = False
                    self.current_step_index = self.current_step_index - 1
                    self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
                    break
                
        if has_error and self.current_step_index < len(self.graph_task) - 1 and self.act_counter[self.current_step_index] > 50 and self.no_act_counter[self.current_step_index] > 10:
            objects_next_step = self.recipe['step_objects'][self.current_step_index + 1]
            for jj in range(len(objects_next_step)):
                h_error = 0
                ele_list = objects_next_step[jj]
                for ele in ele_list:
                    if ele not in detected_objects:
                        h_error = 1
                    elif object_scores[detected_objects.index(ele)] < 0.3:
                        h_error = 1
            
                        
            if h_error == 0:
                has_error = False
                self.current_step_index = self.current_step_index + 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.NEW


        if self.recipe['_id'] == 'pinwheels':
            has_error = self.step_rules_pinwheel(self, detected_objects, object_scores, has_error)
        elif self.recipe['_id'] == 'coffee':
            has_error = self.step_rules_coffee(self, detected_objects, object_scores, has_error)
        elif self.recipe['_id'] == 'mugcake':
            has_error = self.step_rules_mugcake(self, detected_objects, object_scores, has_error)




        ### {'ingredients':[], 'tools': []}
        correct_tools = []
        wrong_tools = []
        wrong_ingredients = []
        correct_ingredients = []

        if has_error:
            all_tools = list(self.recipe['tool_objects'].keys())
            all_ingredients = list(self.recipe['ingredient_objects'].keys())
            main_tools = []
            main_ingredients = []
            for tool_ele in all_tools:
                if tool_ele in object_in_step_set:
                    main_tools.append(tool_ele)
            for ingredient_ele in all_ingredients:
                if ingredient_ele in object_in_step_set:
                    main_ingredients.append(ingredient_ele)

            for tool_ele in main_tools:
                if tool_ele not in detected_objects:
                    wrong_tools.append(tool_ele)
                else:
                    correct_tools.append(tool_ele)
            
            for ingredient_ele in main_ingredients:
                if ingredient_ele not in detected_objects:
                    wrong_ingredients.append(ingredient_ele)
                else:
                    correct_ingredients.append(ingredient_ele)
        
        
        error_entities['ingredients']['right'] = correct_ingredients
        error_entities['ingredients']['wrong'] = wrong_ingredients
        error_entities['tools']['right'] = correct_tools
        error_entities['tools']['wrong'] = wrong_tools

        if len(wrong_ingredients) > 0:
            error_message = f'You are not using the ingredient: {", ".join(wrong_ingredients)}. '

        if len(wrong_tools) > 0:
            error_message += f'You are not using the tool: {", ".join(wrong_tools)}. '

        return has_error, error_message, error_entities

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
                    box_class_score = box['confidence'][o_idx] ** 2 / box['box_confidences'][o_idx]
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

    def _extract_entities(self, step):
        entities = {'ingredients': set(), 'tools': set()}
        tokens, tags = self.recipe_tagger.predict_entities(step)

        for token, tag in zip(tokens, tags):
            if tag == 'INGREDIENT':
                entities['ingredients'].add(token)
            elif tag == 'TOOL':
                entities['tools'].add(token)

        return entities

    def step_rules_pinwheel(self, detected_objects, object_scores, has_error):
        if self.act_counter[self.current_step_index] > 30 and has_error and self.current_step_index == 0:
            if 'Jar of nut butter' in detected_objects and object_scores[detected_objects.index('Jar of nut butter')] > 0.3:
                self.main_step_exe[1][0] = 1
                has_error = False
                self.current_step_index += 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if self.act_counter[self.current_step_index] > 50  and self.current_step_index == 1:
            if 'butter knife' in detected_objects and 'flour tortilla' not in detected_objects and 'Jar of nut butter' not in detected_objects:
                self.main_step_exe[2][0] = 1
                has_error = False
                self.current_step_index += 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if self.act_counter[self.current_step_index] > 50  and self.current_step_index == 2:
            if 'butter knife' in detected_objects and 'Jar of jelly / jam' in detected_objects and object_scores[detected_objects.index('Jar of jelly / jam')] > 0.5:
                self.main_step_exe[3][0] = 1
                has_error = False
                self.current_step_index += 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
            elif 'butter knife' in detected_objects and 'Jar of nut butter' in detected_objects and 'Jar of jelly / jam' not in detected_objects and object_scores[detected_objects.index('Jar of nut butter')] > 0.5:
                has_error = False
                self.current_step_index -= 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
            
        
        if self.act_counter[self.current_step_index] > 50  and self.current_step_index == 3:
            if 'butter knife' in detected_objects and 'flour tortilla' not in detected_objects and 'Jar of jelly / jam' not in detected_objects:
                self.main_step_exe[4][0] = 1
                has_error = False
                self.current_step_index += 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if self.act_counter[self.current_step_index] > 50 and self.current_step_index == 4:
            if 'flour tortilla' in detected_objects and object_scores[detected_objects.index('flour tortilla')] > 0.5:
                self.main_step_exe[5][0] = 1
                has_error = False
                self.current_step_index += 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS

        
        if self.act_counter[self.current_step_index] > 50  and self.current_step_index == 5:
            if 'toothpicks' in detected_objects and object_scores[detected_objects.index('toothpicks')] > 0.3:
                self.main_step_exe[6][0] = 1
                has_error = False
                self.current_step_index += 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        
        if self.act_counter[self.current_step_index] > 50 and has_error and self.current_step_index == 6:
            if 'butter knife' in detected_objects and object_scores[detected_objects.index('butter knife')] > 0.2:
                self.main_step_exe[7][0] = 1
                has_error = False
                self.current_step_index += 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if self.act_counter[self.current_step_index] > 50 and has_error and self.current_step_index == 7:
            if '~12-inch strand of dental floss' in detected_objects and object_scores[detected_objects.index('~12-inch strand of dental floss')] > 0.1:
                self.main_step_exe[8][0] = 1
                has_error = False
                self.current_step_index += 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS

        if self.act_counter[self.current_step_index] > 50 and self.current_step_index in [9, 10]:
            if 'plate' in detected_objects and object_scores[detected_objects.index('plate')] > 0.5:
                self.main_step_exe[11][0] = 1
                has_error = False
                self.current_step_index = 11
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        return has_error
    
    def step_rules_coffee(self, detected_objects, object_scores, has_error):
        if self.act_counter[self.current_step_index] > 50 and self.current_step_index == 0:
            if 'paper basket filter' in detected_objects and len(detected_objects) < 3 and object_scores[detected_objects.index('paper basket filter')] > 0.1:
                self.main_step_exe[2][0] = 1
                has_error = False
                self.current_step_index = 2
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if self.act_counter[self.current_step_index] > 5 and self.current_step_index == 1 :
            if 'paper basket filter' in detected_objects and len(detected_objects) < 3 and object_scores[detected_objects.index('paper basket filter')] > 0.1:
                self.main_step_exe[2][0] = 1
                has_error = False
                self.current_step_index = self.current_step_index + 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if self.act_counter[self.current_step_index] > 100 and self.current_step_index == 2:
            if 'kitchen scale' in detected_objects and object_scores[detected_objects.index('kitchen scale')] > 0.5:
                self.main_step_exe[3][0] = 1
                has_error = False
                self.current_step_index += 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS

        
        if self.act_counter[self.current_step_index] > 100 and self.current_step_index == 3 and has_error:
            if 'coffee grinder' in detected_objects or 'filter cone dripper' in detected_objects:
                has_error = False
                self.current_step_index = self.current_step_index + 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.NEW
        
        if self.act_counter[self.current_step_index] > 100 and self.current_step_index == 4 and has_error:
            if 'thermometer' in detected_objects and object_scores[detected_objects.index('thermometer')] > 0.3:
                self.main_step_exe[5][0] = 1
                has_error = False
                self.current_step_index = self.current_step_index + 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if self.act_counter[self.current_step_index] > 30 and self.current_step_index == 6 and has_error:
            if 'thermometer' in detected_objects:
                has_error = False
                self.current_step_index = self.current_step_index - 1
                self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        return has_error

    def step_rules_mugcake(self, detected_objects, object_scores, has_error):
        if step_ct == 0 and self.act_counter[step_ct] > 30 and has_error and 'small mixing bowl' in detected_objects and object_scores[detected_objects.index('bowl')] > 0.3:
            self.main_step_exe[1][0] = 1
            step_ct += 1
            has_error = False
            self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if step_ct == 1 and self.act_counter[step_ct] > 150 and '2 Tablespoons all-purpose flour' not in detected_objects and '1.5 Tablespoons granulated sugar' not in detected_objects and '1⁄4 teaspoon baking powder' not in detected_objects and 'Pinch salt' not in detected_objects and '1⁄4 teaspoon vanilla extract' not in detected_objects and len(detected_objects) > 0:
            step_ct += 1
            has_error = False
            self.graph_task[self.current_step_index]['step_status'] = StepStatus.NEW
        
        if step_ct == 2 and self.act_counter[step_ct] > 100 and has_error and '2 teaspoons canola or vegetable oil' in detected_objects and object_scores[detected_objects.index('2 teaspoons canola or vegetable oil')] > 0.3:
            self.main_step_exe[3][0] = 1
            step_ct += 1
            has_error = False
            self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if step_ct == 4 and self.act_counter[step_ct] > 100 and has_error and '12-ounce coffee mug' in detected_objects and object_scores[detected_objects.index('12-ounce coffee mug')] > 0.3:
            self.main_step_exe[5][0] = 1
            step_ct += 1
            has_error = False
            self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if step_ct in [4, 5] and self.act_counter[step_ct] > 100 and has_error and 'microwave' in detected_objects and object_scores[detected_objects.index('microwave')] > 0.3:
            self.main_step_exe[6][0] = 1
            step_ct = 6
            has_error = False
            self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        
        if step_ct == 6 and self.act_counter[step_ct] > 100 and has_error and 'toothpicks' in detected_objects and object_scores[detected_objects.index('toothpicks')] > 0.3:
            self.main_step_exe[7][0] = 1
            step_ct += 1
            has_error = False
            self.graph_task[self.current_step_index]['step_status'] = StepStatus.IN_PROGRESS
        return has_error


class RecipeStatus(Enum):
    NOT_STARTED = 'NOT_STARTED'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'


class StepStatus(Enum):
    NOT_STARTED = 'NOT_STARTED'
    NEW = 'NEW'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'
