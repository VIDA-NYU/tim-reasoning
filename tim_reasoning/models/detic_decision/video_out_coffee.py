# import some common libraries
import os
from turtle import st
import numpy as np
import pickle
import cv2
from collections import Counter
import math


#detic_pred = pickle.load(open('/Users/chenzhao/Desktop/ptg-server-ml/data/detic_pred_coffee1.pkl', 'rb'))
egohos_pred = pickle.load(open('/Users/chenzhao/Desktop/ptg-server-ml/data/egohos_pred_coffee_801.pkl', 'rb'))
#egohos_pred2 = pickle.load(open('/Users/chenzhao/Desktop/ptg-server-ml/data/egohos_pred_coffee2_obj2.pkl', 'rb'))
detic_boxes = pickle.load(open('/Users/chenzhao/Desktop/ptg-server-ml/data/detic_pred_coffee_801_all_boxes.pkl', 'rb'))


steps= ['Measure 12 ounces of cold water and transfer the water to a kettle',
        'assemble the filter cone',
        'place the paper filter in the dripper and spread open to create a cone',
        'weigh the coffee beans',
        'grind the coffee beans and transfer the grounds to the filter cone',
        'check the temperature of the water',
        'slowly pour the water over the grounds',
        'Let the coffee drain completely into the mug'
        ]

main_steps = [[1], [0], [0], [0], [0], [1], [0], [0], [0]]

step_tools =[
    [['measuring cup'], ['measuring cup', 'kettle'], ['kettle']],
    [['filter cone dripper']],
    [['brown paper filter'], ['filter cone dripper']], 
    [['kitchen scale'], ['paper bag'], ['kitchen scale', 'paper bag']],
    [['coffee grinder', 'paper bag'], ['coffee grinder'], ['filter cone dripper']],
    [['thermometer'], ['kettle'], ['kettle', 'thermometer'], ['thermometer']],
    [['kettle'], ['kettle', 'filter cone dripper']],
    [['coffee mug']]
    ]
    


step_counter = Counter()
no_act_counter= Counter()
is_exe = [0] * len(steps)
executed = []
tracker_dict = dict()

for ii in range(len(step_tools)):
    exe_st = [0] * len(step_tools[ii])
    executed.append(exe_st)

def run(src, vocab=None, ann_root=None, include=None, exclude=None, out_file=True, fps=10, show=None, **kw):
    """Run multi-target tracker on a particular sequence.
    """
    from ptgprocess.util import VideoInput, VideoOutput, video_feed, draw_boxes, draw_boxes1, get_vocab

    if out_file is True:
        out_file='coffee_final_objects'+os.path.basename(src)
    ct = 0 
    step_ct = 0
    
    no_act_ct = 0
    error_ct = 0
    total = 0
    step_pred = list()
    started = False
    with VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout:
        for i, im in vin:
            xywh = []
            label = []
            full_labels = []
            scores = []
            ego_item = egohos_pred[ct]
            all_boxes = detic_boxes[ct]
            

            box_score_dict = dict()
            score_list = list()
            final_indices = list()
            final_label = list()
            new_all_boxes = list()
            has_added_set = set()
            min_distince = dict()
            for box in all_boxes:
                if ct > 0:
                    #if box['prediction'] in has_added_set:
                    #    continue
                    if box['prediction'] not in tracker_dict:
                        tracker_dict[box['prediction']] = {'indices':[box['indices']], 'score': [box['score'] ** 2 / box['box_score']]}
                        min_distince[box['prediction']] = -1
                        has_added_set.add(box['prediction'])
                    else:
                        box_class_score = box['score'] ** 2 / box['box_score']
                        #if box_class_score < 0.001:
                        #    continue
                        avg_indices = []
                        for iii in range(4):
                            avg_indices.append(np.average([iitem[iii] for iitem in tracker_dict[box['prediction']]['indices']]))
                        avg_score = np.average(tracker_dict[box['prediction']]['score'])
                        
                        dist = np.sqrt(np.abs(avg_indices[0] - box['indices'][0]) ** 2 +  np.abs(avg_indices[1] - box['indices'][1])** 2 + np.abs(avg_indices[2] - box['indices'][2]) ** 2+ np.abs(avg_indices[3] - box['indices'][3])** 2)
                        #if box['prediction'] == 'measuring cup':
                        #    print(dist)
                        #    print(box)
                        
                        if dist > 50:
                            if box_class_score >  avg_score + 0.2:
                                tracker_dict[box['prediction']] = {'indices':[box['indices']], 'score': [box['score'] ** 2 / box['box_score']]}
                                has_added_set.add(box['prediction'])
                                min_distince[box['prediction']] = dist
                            score_list.append(box['score'] ** 2 / box['box_score'])
                        else:
                            if box['prediction'] not in has_added_set:
                                min_distince[box['prediction']] = dist 
                                if len(tracker_dict[box['prediction']]['score']) >= 30:
                                    tracker_dict[box['prediction']]['score'].append(box['score'] ** 2 / box['box_score'])
                                    tracker_dict[box['prediction']]['indices'].append(box['indices'])
                                    tracker_dict[box['prediction']]['indices'].pop(0)
                                    tracker_dict[box['prediction']]['score'].pop(0)
                                else:
                                    tracker_dict[box['prediction']]['score'].append(box['score'] ** 2 / box['box_score'])
                                    tracker_dict[box['prediction']]['indices'].append(box['indices'])
                            else:
                                if dist < min_distince[box['prediction']] - 10 and box['score'] ** 2 / box['box_score'] > tracker_dict[box['prediction']]['score'][-1] - 0.1:
                                    min_distince[box['prediction']] = dist
                                    tracker_dict[box['prediction']]['score'][-1]= box['score'] ** 2 / box['box_score']
                                    tracker_dict[box['prediction']]['indices'][-1] = box['indices']


                            score_list.append(np.average(tracker_dict[box['prediction']]['score']))
                            has_added_set.add(box['prediction'])
                            
                                
                    
                if ct == 0:
                    score_list.append(box['score'] ** 2 / box['box_score'])

                

                idx = str(box['indices'][0]) + '##' +  str(box['indices'][1]) + '##' + str(box['indices'][2]) + '##' + str(box['indices'][3]) + '##'
                if idx not in box_score_dict:
                    box_score_dict[idx] = box['box_score']
                new_all_boxes.append(box)


            sorted_idx = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
            
            #for box in all_boxes:
            if ego_item['label'] != -1:
                started = True
                xx0 = ego_item['indices'][0]
                yy0 = ego_item['indices'][1]
                xx1 = ego_item['indices'][2]
                yy1 = ego_item['indices'][3]
                for b_idx in sorted_idx:
                    item = new_all_boxes[b_idx]
                #for item in detic_pred[ct]:
                    idx = str(item['indices'][0]) + '##' +  str(item['indices'][1]) + '##' + str(item['indices'][2]) + '##' + str(item['indices'][3]) + '##'
                    if ct > 10:
                        if box_score_dict[idx] < 0.3:
                            continue
                        
                        if score_list[b_idx] < 0.05:
                            continue
                    if item['prediction'] in final_label:
                        continue
                    if idx in final_indices:
                        continue


                    x1 = item['indices'][0]
                    y1 = item['indices'][1]
                    x2 = item['indices'][2]
                    y2 = item['indices'][3]
                    if x2 < xx0:
                        continue
                    if x1 > xx1:
                        continue
                    if y2 < yy0:
                        continue
                    if y1 > yy1:
                        continue

                    if x2 - x1 > (x2 - xx0) * 1.5:
                        continue
                    if y2 - y1 > (y2 - yy0) * 1.5:
                        continue
                    if x2 - x1 > (xx1 - x1) * 1.5:
                        continue
                    if y2 - y1 > (yy1 - y1) * 1.5:
                        continue
                    final_label.append(item['prediction'])
                    final_indices.append(idx)
                    xywh.append(np.array(item['indices']))
                    #label.append(item['prediction'] + ' ' + str(box_score_dict[idx])[:4])
                    full_labels.append(item['prediction'] + ' ' + str(score_list[b_idx])[:4])
                    label.append(item['prediction'])
                    scores.append(score_list[b_idx])
                    if ct == 0:
                        tracker_dict[item['prediction']] = {'indices':[item['indices']], 'score': [score_list[b_idx]]}



            xywh1 = []
            label1 = []
            ego_item = egohos_pred[ct]
            
            #print(item)
            if ego_item['label'] != -1:
                label1.append('obj')
                xywh1.append(np.array([ego_item['indices'][0], ego_item['indices'][1], ego_item['indices'][2], ego_item['indices'][3]]))
            

            xxyy = [250, 50]
            if ego_item['label'] == -1:
                #print(no_act_ct, step_ct, started)
                if started:
                    no_act_ct += 1
                    no_act_counter[step_ct] += 1
                #print(no_act_ct, i)
                im = cv2.putText(im, 'no action,  step '  + str(step_ct + 1) , xxyy[:2], cv2.FONT_HERSHEY_SIMPLEX, im.shape[1]/2000, (0, 0, 255), 1)
                if is_exe[step_ct] == 1 and step_ct < len(steps) - 1 and step_ct > 4 and step_counter[step_ct] > 50 and no_act_counter[step_ct] > 10: 
                    step_ct += 1
                    no_act_ct = 0
                    started = False
                elif is_exe[step_ct] == 1 and step_ct < len(steps) - 1 and step_counter[step_ct] > 50 and no_act_counter[step_ct] > 20: 
                    step_ct += 1
                    no_act_ct = 0
                    started = False

            elif len(label) == 0 or ego_item['indices'][2] - ego_item['indices'][0] < 50 or ego_item['indices'][3] - ego_item['indices'][1] < 50:
                im = cv2.putText(im, 'Irrelevant Error! ' + str(step_ct + 1), xxyy[:2], cv2.FONT_HERSHEY_SIMPLEX, im.shape[1]/2000, (0, 0, 255), 1)
            
            else:
                ### map explicit actions first 
                #print('no act counter',step_ct, no_act_counter[step_ct])
                no_act_counter[step_ct] = 0
                step_counter[step_ct] += 1
                has_error = True
                error_idx = []
                for jj, ele_list in enumerate(step_tools[step_ct]):
                    if jj not in main_steps[step_ct]:
                        continue
                    h_error = 0
                    for ele in ele_list:
                        if ele not in label:
                            h_error = 1
                    if h_error == 1:
                        error_idx.append(jj)
                    #all_errors.append(h_error)
                for jj in main_steps[step_ct]:
                    if jj not in error_idx:
                        has_error = False
                        executed[step_ct][jj] = 1
                        break
                
                ### now focus on implicit actions
                if has_error:
                    for jj, ele_list in enumerate(step_tools[step_ct]):
                        if jj in main_steps[step_ct]:
                            continue
                        h_error = 0
                        for ele in ele_list:
                            if ele not in label:
                                h_error = 1
                        if h_error == 0:
                            has_error = False
            
                ##### Here's the heuristic to trace back:
                
                if has_error:
                    exp_exe = False
                    for jj in main_steps[step_ct]:
                        if executed[step_ct][jj] == 1:
                            exp_exe = True
                            break
                    
                    if not exp_exe and step_ct > 0:
                        for jj in range(len(step_tools[step_ct - 1])):
                            ele_list = step_tools[step_ct - 1][jj]
                            h_error = 0
                            for ele in ele_list:
                                if ele not in label:
                                    h_error = 1
                            if h_error == 0:
                                has_error = False
                                step_ct = step_ct - 1
                                break
                    
                if has_error and step_ct < len(steps) - 1 and step_counter[step_ct] > 50 and no_act_counter[step_ct] > 10:
                    ele_list = step_tools[step_ct + 1][0]
                    h_error = 0
                    
                    
                    for jj, ele in enumerate(ele_list):
                        #if len(label) == 1:
                        #    continue

                        if ele not in label:
                            h_error = 1
                        
                        elif scores[label.index(ele)] < 0.3:
                            h_error = 1
                    if h_error == 0:
                        has_error = False
                        step_ct = step_ct + 1
                        no_act_ct = 0
                        started = False
                        #break
                ### I basically ignored action 2 
                if step_counter[step_ct] > 50  and step_ct == 0:
                    if 'brown paper filter' in label and len(label) < 3 and scores[label.index('brown paper filter')] > 0.1:
                        executed[2][0] = 1
                        has_error = False
                        step_ct = 2
                        no_act_ct = 0
                        started = False
                
                if step_counter[step_ct] > 5  and step_ct == 1 :
                    if 'brown paper filter' in label and len(label) < 3 and scores[label.index('brown paper filter')] > 0.1:
                        executed[2][0] = 1
                        has_error = False
                        step_ct = step_ct + 1
                        no_act_ct = 0
                        started = False
                
                if step_counter[step_ct] > 100  and step_ct == 2:
                    if 'kitchen scale' in label and scores[label.index('kitchen scale')] > 0.5:
                        executed[3][0] = 1
                        has_error = False
                        step_ct += 1
                        no_act_ct = 0
                        started = False

                
                if step_counter[step_ct] > 100  and step_ct == 3 and has_error:
                    if 'coffee grinder' in label or 'filter cone dripper' in label:
                        has_error = False
                        step_ct = step_ct + 1
                        no_act_ct = 0
                        started = False
                
                if step_counter[step_ct] > 100  and step_ct == 4 and has_error:
                    if 'thermometer' in label and scores[label.index('thermometer')] > 0.3:
                        executed[5][0] = 1
                        has_error = False
                        step_ct = step_ct + 1
                        no_act_ct = 0
                        started = False
                
                if step_counter[step_ct] > 30  and step_ct == 6 and has_error:
                    if 'thermometer' in label:
                        has_error = False
                        step_ct = step_ct - 1
                        no_act_ct = 0
                        started = False


                im = cv2.putText(im, 'step ' + str(step_ct + 1) + ' ' + steps[step_ct], xxyy[:2], cv2.FONT_HERSHEY_SIMPLEX, im.shape[1]/2000, (0, 0, 255), 1)
                xxyy = [250, 70]
                if has_error:
                    im = cv2.putText(im, 'error', xxyy[:2], cv2.FONT_HERSHEY_SIMPLEX, im.shape[1]/2000, (0, 0, 255), 1)
                    error_ct += 1
                else:
                    im = cv2.putText(im, 'no error', xxyy[:2], cv2.FONT_HERSHEY_SIMPLEX, im.shape[1]/2000, (0, 0, 255), 1)
                    all_finished = True
                    for jj in main_steps[step_ct]:
                        if executed[step_ct][jj] == 0:
                            all_finished = False
                    if all_finished:
                        is_exe[step_ct] = 1


            step_pred.append(step_ct)   
            im2 = draw_boxes(im, xywh, full_labels)
            #im2 = draw_boxes(im, xywh, label)
            #im2 = draw_boxes1(im2, xywh1, label1)
            imout.output(draw_boxes1(im2, xywh1, label1))
            ct += 1

            
            
            
    print(step_counter)  
    print(no_act_counter)   
    print(error_ct, ct, error_ct / ct)   
    print(len(step_pred))
    pickle.dump(step_pred, open('coffee3_step.pkl', 'wb'))

if __name__ == '__main__':
    import fire
    fire.Fire(run)
