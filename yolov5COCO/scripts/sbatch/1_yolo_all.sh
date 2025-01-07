#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM options.
#       Lines starting with "#" and "##SBATCH" are comments.  
#       Uncomment a "##SBATCH" line (i.e. remove one #) to #SBATCH
#       means turn a comment to a SLURM option.

#SBATCH --job-name=1_yolo_all                # Slurm job name
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

srun python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 0 --img-end 10&
srun python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 10 --img-end 20&
srun python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 20 --img-end 30&
srun python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 30 --img-end 40&
srun python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 40 --img-end 50&
srun python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 50 --img-end 60&
srun python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 60 --img-end 70&
srun python main_faithful_cal_adaptive_yolo_optimize_faithfulness_all.py --img-start 70 --img-end 80

wait