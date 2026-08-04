#!/bin/bash
#SBATCH -J mex_test
#SBATCH -p seas_gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0-00:40:00
#SBATCH -o mex_test_%j.out
# Build the MEX interface and run the smoke test (test_mex.m) on a GPU node.
#
# IMPORTANT -- compiler choice: the MEX is loaded into MATLAB's process, and
# MATLAB ships its own libstdc++.  MATLAB R2024b's bundled libstdc++ provides at
# most GLIBCXX_3.4.30, so the MEX must be built with GCC <= 12 (gcc 12 -> 3.4.30).
# Building with gcc 13 links fine but fails at load time with
#   "version `GLIBCXX_3.4.32' not found".
set -e
source /etc/profile.d/lmod.sh
module purge
module load gcc/12.2.0-fasrc01 cuda/12.4.1-fasrc01 cmake/4.2.3-fasrc01 matlab/R2024b-fasrc01
cd "$SLURM_SUBMIT_DIR"
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ls -la build/psd_projection_MATLAB.mexa64
echo "=== GLIBCXX requirement of the built MEX (must be <= 3.4.30) ==="
strings build/psd_projection_MATLAB.mexa64 | grep -oE "GLIBCXX_3\.4\.[0-9]+" | sort -V -u | tail -3
matlab -batch "test_mex"
