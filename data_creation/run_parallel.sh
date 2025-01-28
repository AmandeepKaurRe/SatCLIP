#!/bin/bash
#SBATCH -t 00:30:00                      # wall time (D-HH:MM)
#SBATCH -N 1    
#SBATCH --cpus-per-task=12              # wall time (D-HH:MM)
#SBATCH --mem=25G


#SBATCH -p general
#SBATCH -q public

#SBATCH --output=parallel_%j.log
##SBATCH -A akaur64                    # Account hours will be pulled from (commented out with double # in front)
#SBATCH --mail-type=FAIL                 # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=akaur64@asu.edu    # send-to address


# Load the Conda module
module load mamba/latest

# Activate the Conda environment
source activate satclip_env

cd /home/akaur64/GeoSSL/SATCLIP/data_creation
# Run the Python script
python parallel.py $PYTHON_ARGS