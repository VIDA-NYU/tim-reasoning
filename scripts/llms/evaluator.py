import random
import pandas as pd
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from ego4d_reader import get_goal, aggregate_steps, read_train_data, get_steps, make_general_summary, make_detailed_summary
from llamavid_runner import run

random.seed(27)
annotations = read_train_data()

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


def create_initial_question(video_data):
    goal = get_goal(video_data).lower()
    steps = aggregate_steps(video_data)
    if steps is None:
        return None
    question = f'In the video, the user should be {goal}, these are the recipe steps: {steps}. Summarize the video.'

    return question


def evaluate_offline_video_understanding(video_id):
    model_path = '/scratch/rl3725/llms_experiments/LLaMA-VID/work_dirs/llama-vid/llama-vid-7b-full-224-video-fps-1'
    video_path = f'/scratch/rl3725/llms_experiments/data/ego4d/v2/full_scale/{video_id}_mod.mp4'
    annotations = read_train_data()
    video_data = annotations[video_id]
    general_summary = make_general_summary(video_data)
    detailed_summary = make_detailed_summary(video_data)
    questions = ['Summarize the video.', 'Describe the video in detail.']
    questions = [create_initial_question(video_data), 'Describe the video in detail.']
    question_ids = range(len(questions))
    true_answers = [general_summary, detailed_summary]
    pred_answers = run(model_path, video_path, questions, temperature=0.5)

    print(f'{len(pred_answers)} replied questions.')

    for question_id, true_answer, pred_answer in zip(question_ids, true_answers, pred_answers):
        if questions[question_id] is not None and true_answer is not None:
            #calculate_metrics(question_id, pred_answer, true_answer)
            rouge_score = calculate_rouge(pred_answer, true_answer)
            bleu_score = calculate_bleu(pred_answer, true_answer)
            yield {'ROUGE': rouge_score, 'BLEU': bleu_score}


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


def eval_baseline(video_ids, baseline_method):
    for i, video_id in enumerate(video_ids, 1):
        video_alias = f'video_{i}'
        video_data = annotations[video_id]
        general_summary = make_general_summary(video_data)
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

        for question_id, true_answer, pred_answer in zip(question_ids, true_answers, pred_answers):
            if questions[question_id] is not None and true_answer is not None:
                rouge_score = calculate_rouge(pred_answer, true_answer)
                bleu_score = calculate_bleu(pred_answer, true_answer)
                yield {'video': video_alias, 'ROUGE': rouge_score, 'BLEU': bleu_score}


def evaluate_all_videos(video_ids):
    results = {'ROUGE': {}, 'BLEU': {}}
    methods = ['ALL_STEPS', 'HALF_STEPS', 'ONE_STEP', 'CANTINFLAS']
    
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
    print(pd.DataFrame.from_dict(rouge_dict))
    print('BLEU')
    print(pd.DataFrame.from_dict(bleu_dict))
            

if __name__ == '__main__':
    video_ids = ['1938c632-f575-49dd-8ae0-e48dbb467920', '51224e32-3d6c-4148-9eea-7b73da751f25',
                 '0c192ca8-1ede-4ef0-a05e-2f4151b6bdfc', 'ac582760-09b1-4a6e-be08-f19f9bf5dfcb', 
                 'grp-42686a5b-10d2-499f-b9a8-8043f528efdd']
    
    evaluate_all_videos(video_ids)