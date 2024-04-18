#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem 48GB
#SBATCH --time 00:30:00
#SBATCH --job-name=llms
#SBATCH --output logs/video_processor_job_%J.out


./sing << EOF

cd LLaMA-VID/
python video_processor.py ${1}

EOF
