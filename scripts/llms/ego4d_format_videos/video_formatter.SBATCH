#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32GB
#SBATCH --time 01:00:00
#SBATCH --output logs/videoformatter_job_%J.out
#SBATCH --mail-user=rl3725@nyu.edu

./sing << EOF

ffmpeg -i /scratch/rl3725/llms_experiments/data/ego4d/v2/full_scale/${1}.mp4 -c:v libx264 -crf 23 -preset medium -r 10 -c:a copy  /scratch/rl3725/llms_experiments/data/ego4d/v2/full_scale_mod/${1}.mp4

EOF