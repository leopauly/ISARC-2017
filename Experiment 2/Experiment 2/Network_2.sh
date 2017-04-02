#$ -cwd -V
#$ -l h_rt=10:00:00
#$ -pe smp 24
#$ -l h_vmem=3G
#$ -m be
#$ -M cnlp@leeds.ac.uk
module load singularity
singularity exec /nobackup/containers/ds1.img python < raul_5.py >> raul_5_2.txt
