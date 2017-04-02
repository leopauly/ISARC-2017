#$ -cwd -V
#$ -l h_rt=10:00:00
#$ -pe smp 20
#$ -l h_vmem=3G
#$ -m be
#$ -M cnlp@leeds.ac.uk
module load singularity
singularity exec /nobackup/containers/ds1.img python < Network_1_1.py >> Network_1_1.txt
