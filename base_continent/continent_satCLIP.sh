#!/bin/bash

#SBATCH -N 1
#SBATCH -n 10
#SBATCH -t 3-00:00:00                      # wall time (D-HH:MM)
#SBATCH --mem=256G 

#SBATCH --ntasks-per-node=10

#SBATCH -p general
#SBATCH -q public
#SBATCH -G a100:1

#SBATCH -o logs/continent_aman.out                    # STDOUT (%j = JobId)
#SBATCH -e logs/continent_aman.err                    # STDERR (%j = JobId)

##SBATCH -A akaur64                    # Account hours will be pulled from (commented out with double # in front)
#SBATCH --mail-type=FAIL                 # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=akaur64@asu.edu    # send-to address


# # Load the Conda module
# module load anaconda3

# # Activate the Conda environment
# conda activate satclip
module load mamba/latest

source activate satclip_new

# Run the Python script
python /home/akaur64/GeoSSL/SATCLIP/satclip/satclip/continent_main.py 