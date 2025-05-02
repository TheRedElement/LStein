#!/bin/bash
#SBATCH --mail-type=NONE #ALL
##SBATCH --mail-user=lukas.steinwender99@gmail.com

#SBATCH --job-name=myjob     #job name

#SBATCH --array=0-2                     #slurm array to execute multiple jobs at once
#SBATCH --output=./execlogs/%x_%a.out
#SBATCH --error=./execlogs/%x_%a.err

#SBATCH --partition=icg                 #partition to be used
#SBATCH --nodes 1                       #number of nodes to be used
#SBATCH --ntasks=1                      #number of tasks per node
#SBATCH --gpus-per-node=1               #number of gpus to be used per node
#SBATCH --cpus-per-gpu=4                #number of cpus of each gpu to be used
#SBATCH --time=2-00:00:00               #time limit
#SBATCH --tmp=150GB                     #temporary memory (if large files are acessed, loads of files are read and write)

#################
#start of script#
#################
#print current date
start=$(date +%s)   #start date in seconds
startdate=$(date -u -d "@$start" +%Y-%m-%dT%H:%M:%S)    #format nicely
printf "BASH.INFO: Started at %s\n" $startdate

##############################
#custom environment variables#
##############################

#julia custom depot path
##NOTE: https://docs.julialang.org/en/v1/base/constants/#Base.DEPOT_PATH
##NOTE: https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_DEPOT_PATH
export JULIA_DEPOT_PATH="/path/to/juliadepot:"
# unset JULIA_DEPOT_PATH         #reset to default (if needed)

#########
#control#
#########

slurm_array_length=$((SLURM_ARRAY_TASK_MAX-SLURM_ARRAY_TASK_MIN+1))     #length of slurm job array (set to 1 is not a slurm job)

######
#main#
######

#copy large files to temporary directory
cp /path/to/file.txt $JOBFS

#do stuff...

cp $JOBFS/path/to/output.txt /path/to/target/directory


#julia (needs absolute reference to julia executable)
$HOME/julia-1.10.4/bin/julia --project=~/<path2venv> -t 1 -p 1 < path2file.jl >  #-t ... number of threads #-p ... number of processes

#python
source ~/<path2env>/bash/bin/activate
python3 <path2file.py>
deactivate

#apptainer
apptainer run -B <directory2bind/> <path2container.sif> python3 <path2file.py>  #python
apptainer run -B <directory2bind/> --nv <path2container.sif> python3 <path2file.py>  #python + gpu

###############
#end of script#
###############
finish=$(date +%s)  #end date in seconds
finishdate=$(date -u -d "@$finish" +%Y-%m-%dT%H:%M:%S)  #format nicely

#elapsed time
days="$(($(($finish - $start )) / 86400))"  #elapsed days (convert seconds to days)
duration="$(($finish - $start))"            #get elapsed time in seconds
printf "BASH.INFO: Finished at %s\n" $finishdate
printf "BASH.INFO: Ran from : %s\n" "$startdate to $finishdate"
printf "BASH.INFO: Elapsed time: %s days %s\n" "$days" "$(date -u -d "@$duration" +%H:%M:%S)"

