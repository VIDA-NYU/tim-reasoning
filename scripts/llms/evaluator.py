import sys
import os
import random
import numpy as np
import pandas as pd
from os.path import dirname, join
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from ego4d_reader import get_goal, aggregate_steps, get_valid_annotations, get_steps, make_general_summary, \
    make_detailed_summary, get_summary_sentences
#from llamavid_runner import run

random.seed(27)
annotations = get_valid_annotations()


def calculate_bleu(pred, target):
    bleu = BLEUScore(n_gram=1)
    score = bleu([pred], [[target]]).item()
    return round(score, 3)


def calculate_rouge(pred, target):
    rouge = ROUGEScore()
    score = rouge(pred, target)['rouge1_fmeasure'].item()
    return round(score, 3)


def calculate_metrics(question_id, pred, target):
    rouge_score = calculate_rouge(pred, target)
    bleu_score = calculate_bleu(pred, target)
    print(f'Q{question_id}:  BLEU: {bleu_score}   ROUGE: {rouge_score}')


def create_question_with_context(video_data):
    goal = get_goal(video_data).lower()
    steps = aggregate_steps(video_data)
    if steps is None:
        return None
    question = f'In the video, the user should be {goal}, these are the recipe steps: {steps}. Summarize the video.'

    return question


def run_llamavid(video_ids, question_mode='with_context'):
    results = {'video_id': [], 'mode': [], 'answer': []}
    output_path = 'results_llamavid.csv'

    for video_id in video_ids:
        model_path = '/scratch/rl3725/llms_experiments/LLaMA-VID/work_dirs/llama-vid/llama-vid-7b-full-224-video-fps-1'
        video_path = f'/scratch/rl3725/llms_experiments/data/ego4d/v2/full_scale_mod/{video_id}.mp4'
        video_data = annotations[video_id]
        if question_mode == 'with_context':
            questions = [create_question_with_context(video_data)]
        elif question_mode == 'without_context':
            questions = ['Summarize the video.']

        pred_answers = run(model_path, video_path, questions, temperature=0.5)

        results['video_id'].append(video_id)
        results['mode'].append(question_mode)
        results['answer'].append(pred_answers[0])
    
    results_df = pd.DataFrame.from_dict(results)
    exist_file = os.path.exists(output_path)
    results_df.to_csv(output_path, index=False, mode='a', header=not exist_file)


def make_baseline1_summary(video_data, num_steps='all'):
    steps = get_steps(video_data)

    if num_steps == 'all':
        pass
    elif num_steps == 'half':
        steps = random.choices(steps, k=len(steps)//2)
    elif num_steps == 'one':
        steps = random.choices(steps, k=1)

    dummy_summary = ' '.join(steps)

    return dummy_summary


def make_baseline2_summary():
    dummy_summaries = ['The video captures the essence of culinary artistry as a person gracefully navigates through the kitchen, orchestrating a symphony of flavors and textures without delving into specifics. With a blend of skill and intuition, they seamlessly execute each step of the cooking process, captivating the viewer with their expertise. The ambiance is filled with an air of excitement and anticipation as the culinary masterpiece takes shape, leaving the audience intrigued by the creative process.',
                       'Amidst the hustle and bustle of the kitchen, the video showcases a person culinary journey with a focus on technique and creativity. Each movement is deliberate and purposeful, reflecting a deep understanding of the intricacies of cooking. Despite the absence of ingredient details, the viewer is drawn into the captivating world of culinary craftsmanship, where passion and skill converge to create an unforgettable dining experience.',
                       'With finesse and precision, the video captures the art of cooking as a person effortlessly navigates through the culinary landscape, crafting a delicious dish without explicit mention of ingredients or utensils. The atmosphere is imbued with a sense of joy and satisfaction as the culinary creation unfolds, leaving the viewer inspired by the beauty of the cooking process. In the absence of specifics, the focus remains on the sheer artistry and skill displayed in bringing the dish to life.']
    dummy_summary = random.choices(dummy_summaries, k=1)[0]

    return dummy_summary


def make_perfect_summary(video_data):
    steps = get_summary_sentences(video_data)

    if steps is None:
        return None
    
    perfect_summary = ' '.join(steps)

    return perfect_summary


def eval_baseline(video_ids, baseline_method):
    for i, video_id in enumerate(video_ids, 1):
        video_alias = f'video_{i}'
        video_data = annotations[video_id]
        general_summary = make_general_summary(video_data)
        general_summary = create_question_with_context(video_data)
        questions = ['Summarize the video.']
        question_ids = range(len(questions))
        true_answers = [general_summary]

        if baseline_method == 'ALL_STEPS':
            pred_answers = [make_baseline1_summary(video_data, 'all')]
        elif baseline_method == 'HALF_STEPS':
            pred_answers = [make_baseline1_summary(video_data, 'half')]
        elif baseline_method == 'ONE_STEP':
            pred_answers = [make_baseline1_summary(video_data, 'one')]
        elif baseline_method == 'CANTINFLAS':
            pred_answers = [make_baseline2_summary()]
        elif baseline_method == 'PERFECT':
            pred_answers = [make_perfect_summary(video_data)]

        for question_id, true_answer, pred_answer in zip(question_ids, true_answers, pred_answers):
            rouge_score = calculate_rouge(pred_answer, true_answer)
            bleu_score = calculate_bleu(pred_answer, true_answer)
            yield {'video': video_alias, 'ROUGE': rouge_score, 'BLEU': bleu_score}


def evaluate_all_baselines(video_ids):
    results = {'ROUGE': {}, 'BLEU': {}}
    methods = ['ALL_STEPS', 'HALF_STEPS', 'ONE_STEP', 'CANTINFLAS', 'PERFECT']
    
    for method_name in methods:
        if method_name not in results:
            results['ROUGE'][method_name] = {}
            results['BLEU'][method_name] = {}
        for result_eval in eval_baseline(video_ids, method_name):
            results['ROUGE'][method_name][result_eval['video']] = result_eval['ROUGE']
            results['BLEU'][method_name][result_eval['video']] = result_eval['BLEU']
    
    rouge_dict = {k: v.values() for k, v in results['ROUGE'].items()}
    bleu_dict = {k: v.values() for k, v in results['BLEU'].items()}
    print('ROUGE')
    #print(pd.DataFrame.from_dict(rouge_dict))
    for baseline, scores in rouge_dict.items():
        scores = list(scores)
        print(f'{baseline} {round(np.mean(scores), 3)}  ±{round(np.std(scores), 3)}')

    print('BLEU')
    #print(pd.DataFrame.from_dict(bleu_dict))
    for baseline, scores in bleu_dict.items():
        scores = list(scores)
        print(f'{baseline} {round(np.mean(scores), 3)}  ±{round(np.std(scores), 3)}')


def evaluate_llamavid(video_ids, question_mode):
    results = pd.read_csv(join(dirname(__file__), 'llamavid_results.csv'))
    results = results[results['mode'] == question_mode]
    results.set_index('video_id', inplace=True)
    results.fillna('', inplace=True)
    print(f'Found {len(results)} videos')
    results = results.to_dict()
    rouge_scores = []
    bleu_scores = []

    for video_id in video_ids:
        video_data = annotations[video_id]
        true_answer = make_general_summary(video_data)
        true_answer = create_question_with_context(video_data)
        pred_answer = ''

        if video_id in results['answer']:
            pred_answer = results['answer'][video_id]
        else:
            print('No answer found')

        rouge_score = calculate_rouge(pred_answer, true_answer)
        bleu_score = calculate_bleu(pred_answer, true_answer)
        rouge_scores.append(rouge_score)
        bleu_scores.append(bleu_score)
    
    print(f'ROUGE {round(np.mean(rouge_scores), 3)}  ±{round(np.std(rouge_scores), 3)}')
    print(f'BLEU {round(np.mean(bleu_scores), 3)}  ±{round(np.std(bleu_scores), 3)}')
    
    #print(dataset) 

if __name__ == '__main__':
    video_ids = ['39d087b0-afc2-47d8-ba91-b70dd8fab90e', '2f6da5f6-e26c-4ac3-8f71-12386f7588e2', 
                 '09bccca1-368b-4776-9433-3c8835837110', '6639f53c-701d-4fe7-adcd-55d040ce8afe']
    
    video_ids = ['39d087b0-afc2-47d8-ba91-b70dd8fab90e', '2f6da5f6-e26c-4ac3-8f71-12386f7588e2', '09bccca1-368b-4776-9433-3c8835837110', '6639f53c-701d-4fe7-adcd-55d040ce8afe', 'grp-090c6bc0-49da-4d3b-b209-a1a60aeb0317', 'd9691bde-a0b2-4521-8374-a74f594aaaac', '4c642620-db0e-4096-9ece-2b2c6fdb47b0', 'grp-ab070b36-def2-4ad7-a760-2a9ce29ce505', 'grp-d250521e-5197-44aa-8baa-2f42b24444d2', '5cdf77b8-7bf8-421b-99b6-19fa6429aeb4', '546a1aed-676d-44ed-a63c-8db89fa4d935', '94d5eff8-0fac-4719-adf2-5c0208ab89f7', '91e3e6ce-bc01-4720-a490-e319dd380509', '1938c632-f575-49dd-8ae0-e48dbb467920', 'b0a55292-f9a0-4af2-8f2a-bd6d7eff2b2e', '51224e32-3d6c-4148-9eea-7b73da751f25', 'b83285c5-0b88-4ced-a52e-5c34ea371507', 'grp-2be0151d-8ee7-4e7a-adf5-6b2b3d5afdb0', '68205f0f-9b30-4d81-986d-f8816e70bddd', '0ebb682e-6aec-482c-bbb7-774ec5eca906', '78b06017-cd7b-441c-ba48-33c04e37a82a', '1bece8d5-2d0f-47bb-a1f0-4fc3f94970f5', '4fa75795-ddc4-4582-9715-bb7887439263', 'grp-344098b8-3b27-4a98-a11d-f18fa5c25a5c', 'grp-5c00fc7f-e94b-47ac-98bb-a04896fd6092', 'grp-0cf6e8bd-48a6-4764-9367-c59603a00e4e', 'c939aae4-e7c8-453c-8671-40573db0c656', 'f5b87541-3879-444f-be65-2b17f1d95735', '8383791d-2df9-4bac-89b6-4cc01df5f4d1', '748536e4-636a-4dc6-b1a7-d9cbfdc1cffd', 'grp-66a0900b-069d-48bf-9071-f4e659f8a9d0', 'f7a0beb6-b220-40c0-a72d-ec4b79134a73', 'grp-9f28e782-417c-4c8b-a7ae-42fc96a0e94f', 'grp-e191e0de-e570-4925-9cbb-e05fe1132a47', '4bddae9e-8ffb-4a03-9421-adf6268d91b6', '48475a9f-e11e-46f7-bdb6-398b44af344e', 'b4072935-56a6-4765-bb4d-d5f6bbeb95b9', 'aa7e4a70-abda-4ef4-a1e0-5af5bda1e560', '737e9619-7768-407c-8a4f-6fe1e8d61f04', 'c79e05d5-eaf6-4323-a707-744020743037', '0fe191ef-c28a-422c-aede-46f8aa8532a6', '1ab9d5f7-0181-458e-a5e7-72ce87501f3e', '71c04f59-46c2-4ac8-af56-e04b29acc8ec', 'grp-0eb88d90-6da1-4957-bc4f-1ced804cd7ce', '1025f75a-dc3d-4a90-9542-8a92755e0761', '9961615a-7e54-4b22-84ee-8e05f52c22df', '97811639-7def-4034-8083-a82a59156234', 'grp-0b215326-8706-4611-9b76-3402d617f19e', '2c27b5f1-4af6-49ad-a43c-3efb0c150868', 'ec3556de-be79-4ad4-aa0f-eaca48abb5d5', 'd51a0b73-1ad7-4d4e-be99-714f6637668a', '26202090-684d-4be8-b3cc-de04da827e91', 'grp-bbd4830e-fad8-4339-8290-eb298cf6ca30', 'f4cc5fdc-f64f-4dd7-9b95-61db9bbf33d5', 'grp-fc70c4fe-13e1-4691-ab97-501a22b3a71e', '3a03f541-a520-4b67-87b2-f9c79f9b54ee', 'e7bb40bd-2ba3-4ac9-8e10-b209504f2964', 'grp-a738fe92-4d8e-48db-9c1f-280a7471b5dd', 'fa35a547-13b5-4355-873c-322c1860349a', 'grp-6fea409b-159e-45cb-8fac-da7779d02309', 'grp-94cdabf3-c078-4ad4-a3a1-c42c8fc3f4ad', 'f42ae9c2-d43e-45af-8adb-02d0b16b2ef7', '767dbef9-f625-4785-810c-7d8c78c1eddb', 'be8889c4-114f-4cb2-9e2c-fef576dbb00d', 'f938bcd9-bf30-4dfb-9a99-d6b9ee53c046', 'grp-86b7a8dc-e0ba-4b21-a097-eed463fb2747', '15d2dc4a-b935-49f4-ae91-4c9100b681c8', '2f46d1e6-2a85-4d46-b955-10c2eded661c', 'grp-f6a0d8f5-468d-4e47-bfca-e7d13b75f26d', 'cda837dd-1632-45f8-88f7-cfbf7875523b', 'ebc9f0f8-924b-47d1-a639-de4f2d6e6a4c', '3ec3eab7-842d-409d-8866-42ddcbd24cd9', '37d183ce-ba64-4ad2-8a54-ae2fb791fd38', '272e8bcd-32a8-48ce-a9c4-a5f0a4f15145', 'ab875220-cd6c-410d-8fbc-94742e51e775', 'grp-65403036-df60-4ccb-ac72-2808b841665c', 'grp-a3e528b8-dce9-431f-b3ca-b2707f3439d2', '93f78437-229d-4824-a8c1-68d9b17e754d', '42d548dc-bb10-427f-a1fe-468f31bc0de5', 'grp-6de30946-5d3a-47a8-b1d0-790937405da3', 'grp-d13aa822-29de-428a-bdf4-f4d6ec4a22e0', 'grp-745aef91-750a-48aa-9464-88802a2a7a40', 'grp-d59f69bd-8250-4d06-998b-f057f51e6a8f', 'grp-415c9b83-b4a1-49ba-832b-f77c2e6f996c', '42e4a840-68a1-4992-923d-7452500b4218', 'grp-95710114-4168-47b4-a63e-a4d220b42fcf', 'grp-ef725741-fe17-4bd2-ad6e-aba3e9468fd5', '049cfb89-3bcc-4fa3-8d0e-4e7e218b05ae', 'f43bb351-fb98-4a74-bd50-f57e91d3863f', '5093842e-7cf9-4509-bf04-6e0ec6b75b27', 'grp-530dd2a8-462c-450b-9f38-1dfff28f139e', 'grp-87cfce50-3d03-4ffe-b9a7-e94a435175e3', 'grp-5aa3cd50-4925-4a1a-a8d5-2e18af017660', 'f735d4bb-d65e-4965-ac0d-270c0b9e5993', '4474c9c2-9d20-46fd-8fec-7d64942738e7', 'e3d02ddb-f386-4ae3-a62d-e4fe4db7f345', 'bc6bf18b-0ef8-4cb9-aa51-d5d3558916e5', '690f58f1-f18c-4415-bab0-787c2f83d051', 'grp-229e8124-605e-4c86-b4ac-fc03e0a71111', 'grp-dc4afc9e-7a2d-4876-b5cf-fde740582358']
    
    #evaluate_all_baselines(video_ids)
    #video_ids = [sys.argv[1]]
    question_mode = 'with_context'
    #run_llamavid(video_ids, question_mode)
    evaluate_llamavid(video_ids, question_mode)
