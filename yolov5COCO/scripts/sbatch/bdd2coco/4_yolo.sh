#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM options.
#       Lines starting with "#" and "##SBATCH" are comments.  
#       Uncomment a "##SBATCH" line (i.e. remove one #) to #SBATCH
#       means turn a comment to a SLURM option.

#SBATCH --job-name=4_yolo                # Slurm job name
#SBATCH --time=3-00:00:00                    # Set the maximum runtime
#SBATCH --partition=gpu-a30                  # Choose partition
#SBATCH --account=abclab            # Specify project account


# Resource allocation 

#SBATCH --nodes=1                      # node count
#SBATCH --ntasks-per-node=1            # number of tasks per node (adjust when using MPI)
#SBATCH --cpus-per-gpu=16              # cpu-cores per task (>1 if multi-threaded tasks, adjust when using OMP)
#SBATCH --gpus-per-node=1              # Number of GPUs for the task

# Email notificaitons

#SBATCH --mail-user=shaderein@hotmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE


### Run vanilla saliency maps. COCO finished on division server
eval "$(conda shell.bash hook)"
conda activate xai-3.8

# Go to the job submission directory and run your application
cd $HOME/xai/yolov5COCO

for category in vehicle human; do

    python main_faithful_cal_adaptive_yolo_optimize_faithfulness_bdd2coco.py --object $category --img-start 120 --img-end 160 --device 0

done