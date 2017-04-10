#!/bin/bash
#SBATCH --job-name=find_dips
#SBATCH --output=LOGS/%a_%A_slurm.out # %A: master job ID, %a: array tasks ID.
#SBATCH --array=1-198
#SBATCH -N 1   # node count. OpenMP requires 1.
#SBATCH --ntasks-per-node=1  # core count.
#SBATCH -t 3:15:00 # 5min gets test queue. e.g., 3:00:00 for 3hr
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=end
#SBATCH --mail-user=lbouma@princeton.edu

# Load the appropriate venv

module load anaconda3
source activate cbp

# $SLURM_ARRAY_TASK_ID is the "environmental" variable slurm uses to index
# the job array. We'll use it to directly index our list of KIC IDs.

linenumber=$SLURM_ARRAY_TASK_ID
kicid="$(sed "${linenumber}q;d" ../data/not_done_ids.txt)"

srun python run_the_machine.py -frd -c -kicid $kicid -nw 1 > LOGS/"$linenumber"_"$kicid".log
