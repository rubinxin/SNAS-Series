#!/bin/bash
    ##############################################################
    
    ##############################################################
    
# set the number of nodes
#SBATCH --partition=small

# set the number of nodes
#SBATCH --nodes=1

# set the number of GPU cards to use per node
#SBATCH --gres=gpu:1

# set max wallclock time
#SBATCH --time=72:00:00

# set name of job
#SBATCH --job-name=norcdarts1
#SBATCH --output=norcdarts1.out

# mail alert at start and end of execution
#SBATCH --mail-type=ALL

cd /jmain01/home/JAD017/sjr02/rxr89-sjr02/SNAS-SeriesDarts/SNAS/
module load python3/anaconda
source activate gpy
cd /jmain01/home/JAD017/sjr02/rxr89-sjr02/SNAS-SeriesDarts/SNAS/


# run the application
python train_search.py --snas --epochs 150 --seed 6 --layer 8 --init_channels 16 --temp 1 \
--temp_min 0.03 --nsample 1 --temp_annealing --gen_max_child \
--resource_lambda 1.5e-3 --log_penalty --drop_path_prob 3e-1 --method 'reparametrization' \
--loss --remark "snas_darts_norc_layer_8_batch_64_drop_0.3_error_lnR_1e-2_reparam_gpu_1"
        
