#!/bin/bash
#SBATCH --account=def-florian7
#SBATCH --job-name=room-parallel2
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --output=%x-%j.out

export VENV=~/venv
export SPACE_DIR=~/SPACE

module load python/3.7
#virtualenv --no-download $VENV/
source $VENV/bin/activate
#pip install --no-index --upgrade pip
#pip install attrdict
#pip install --no-index -r $SPACE_DIR/requirements_cc.txt

#pip install torch==1.3.0
#pip install torchvision==0.4.0

echo 'Unzipping data'

mkdir $SLURM_TMPDIR/data
tar -xzf ~/projects/def-florian7/zqallan/OBJ3D_SMALL.tar.gz -C $SLURM_TMPDIR/data
#tree $SLURM_TMPDIR/data

date
echo 'Training...'

cd $SPACE_DIR/src
python main.py --task train --config configs/3d_room_small.yaml resume True parallel True device 'cuda:0' device_ids '[0, 1]'


