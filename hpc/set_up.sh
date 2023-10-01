#!/bin/bash
#SBATCH --job-name=GPUOnFox
#SBATCH --account=ec232
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=512M
#SBATCH --qos=devel
#SBATCH --partition=accel
#SBATCH --gpus=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module purge
module load JupyterLab/3.5.0-GCCcore-11.3.0
source /fp/projects01/ec232/venvs/in5310/bin/activate

# Use 'which' to find the path to Python and print it
python_path=$(which python)
echo "Python executable path: $python_path"

## python p2_01_test01.py

