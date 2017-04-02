#$ -cwd -V
#$ -l h_rt=2:00:00
#$ -pe smp 20
#$ -l h_vmem=3G
#$ -m be
#$ -M cnlp@leeds.ac.uk
module load singularity
singularity exec /nobackup/containers/ds1.img python < experiment_2_2.py >> experiment_2_2_results.txt
