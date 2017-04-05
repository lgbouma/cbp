#!/bin/bash
#SBATCH --job-name=find_dips
#SBATCH --output=LOGS/%a_%A_slurm.out # %A: master job ID, %a: array tasks ID.
#SBATCH --array=1-10
#SBATCH -N 1   # node count. OpenMP requires 1.
#SBATCH --ntasks-per-node=2  # core count. 2 per star, for testing.
#SBATCH -t 0:04:30 # 5min gets test queue. e.g., 3:00:00 for 3hr
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=lbouma@princeton.edu

# $SLURM_ARRAY_TASK_ID is the "environmental" variable slurm uses to index
# the job array. We'll use it to directly index our list of KIC IDs.

linenumber=$SLURM_ARRAY_TASK_ID
kicid="$(sed "${linenumber}q;d" ../data/morph_gt_0.6_ids.txt)"

srun python run_the_machine.py -frd -c -kicid $kicid > LOGS/"$linenumber"_"$kicid".log
