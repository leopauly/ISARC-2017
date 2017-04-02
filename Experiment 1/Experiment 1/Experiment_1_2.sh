#$ -cwd -V
#$ -l h_rt=03:00:00
#$ -pe smp 12
#$ -l h_vmem=10G
#$ -m be
#$ -M cnlp@leeds.ac.uk
module load singularity
singularity exec /nobackup/containers/ds1.img python < Experiment_1_2.py >> Experiment_1_2.txt
