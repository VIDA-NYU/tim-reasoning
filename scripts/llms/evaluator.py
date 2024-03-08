from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from ego4d_reader import *
from llamavid_runner import run

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
            calculate_metrics(question_id, pred_answer, true_answer)


if __name__ == '__main__':
    videos = ['1938c632-f575-49dd-8ae0-e48dbb467920', '51224e32-3d6c-4148-9eea-7b73da751f25',
              '0c192ca8-1ede-4ef0-a05e-2f4151b6bdfc', 'ac582760-09b1-4a6e-be08-f19f9bf5dfcb', 
              'grp-42686a5b-10d2-499f-b9a8-8043f528efdd']
    
    evaluate_offline_video_understanding(videos[4])