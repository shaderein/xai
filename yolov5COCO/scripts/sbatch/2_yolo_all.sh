#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM options.
#       Lines starting with "#" and "##SBATCH" are comments.  
#       Uncomment a "##SBATCH" line (i.e. remove one #) to #SBATCH
#       means turn a comment to a SLURM option.

#SBATCH --job-name=2_yolo_all                # Slurm job name
#SBATCH --time=3-00:00:00                    # Set the maximum runtime
#SBATCH --partition=gpu-a30                  # Choose partition
#SBATCH --account=abclab            # Specify project account


# Resource allocation 

#SBATCH --ntasks=8                           # Total number of tasks
#SBATCH --cpus-per-task=8                    # 8 CPU cores per task
#SBATCH --gpus-per-task=1 

# Email notificaitons

#SBATCH --mail-user=shaderein@hotmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

### Run vanilla saliency maps. COCO finished on division server
eval "$(conda shell.bash hook)"
conda activate xai-3.8

# Go to the job submission directory and run your application
cd $HOME/jinhan/xai/yolov5COCO

srun --ntasks=1 python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 80 --img-end 90&
srun --ntasks=1 python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 90 --img-end 100&
srun --ntasks=1 python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 100 --img-end 110&
srun --ntasks=1 python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 110 --img-end 120&
srun --ntasks=1 python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 120 --img-end 130&
srun --ntasks=1 python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 130 --img-end 140&
srun --ntasks=1 python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 140 --img-end 150&
srun --ntasks=1 python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 150 --img-end 160

wait