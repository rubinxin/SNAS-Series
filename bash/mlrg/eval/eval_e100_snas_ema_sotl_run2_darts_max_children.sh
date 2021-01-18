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
#SBATCH --job-name=evalemasotldarts2e100
#SBATCH --output=evalemasotldarts2e100.out

# mail alert at start and end of execution
#SBATCH --mail-type=ALL

cd /jmain01/home/JAD017/sjr02/rxr89-sjr02/SNAS-SeriesDarts/SNAS/
module load python3/anaconda
source activate gpy
cd /jmain01/home/JAD017/sjr02/rxr89-sjr02/SNAS-SeriesDarts/SNAS/


# run the application
python train.py --auxiliary --cutout --seed 7 --arch 'snas_emasotl_darts_run2_epoch100'

