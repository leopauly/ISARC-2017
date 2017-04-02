#$ -cwd -V
#$ -l h_rt=15:00:00
#$ -pe smp 18
#$ -l h_vmem=5G
#$ -m be
#$ -M cnlp@leeds.ac.uk
module load singularity
singularity exec /nobackup/containers/ds1.img python < Network_2.py >> Network_2.txt
