#!/bin/bash
#SBATCH --job-name=injrecov
#SBATCH --output=LOGS/%a_%A_slurm.out # %A: master job ID, %a: array tasks ID.
#SBATCH --array=1-1000 # starting at 1, out to last index of /data/N_to_KICID.txt
#SBATCH -N 1   # node count. OpenMP requires 1.
#SBATCH --ntasks-per-node=2  # core count.
#SBATCH -t 11:59:00 # 1hr min for fast queue. e.g., 3:00:00 for 3hr
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=end
#SBATCH --mail-user=lbouma@princeton.edu

# Load the appropriate venv

module load anaconda3
source activate cbp

# $SLURM_ARRAY_TASK_ID is the "environmental" variable slurm uses to index
# the job array. We'll use it to directly index our list of KIC IDs.

# FIXME: linenumber should have better logic
linenumber=$((7000+$SLURM_ARRAY_TASK_ID))

N=$(($linenumber-1))

kicid="$(sed "${linenumber}q;d" /tigress/lbouma/data/N_to_KICID.txt | awk '{print $2}')"

srun python run_the_machine.py --injrecov -c -kicid $kicid -nw 2 --Nstars $N -mp > LOGS/"$linenumber"_"$kicid".log
