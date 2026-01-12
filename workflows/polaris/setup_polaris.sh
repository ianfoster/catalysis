#!/bin/bash
# Setup script for Catalyst on Polaris
#
# Usage:
#   # Copy this script to Polaris and run:
#   bash setup_polaris.sh
#
# Or run directly:
#   ssh polaris.alcf.anl.gov 'bash -s' < workflows/polaris/setup_polaris.sh

set -e

echo "=========================================="
echo "Catalyst Setup for Polaris"
echo "=========================================="

# Configuration - edit these as needed
EAGLE_PROJECT="/eagle/AuroraGPT/foster"
CATALYST_DIR="${EAGLE_PROJECT}/catalysis"
CONDA_DIR="${EAGLE_PROJECT}/conda"
ALLOCATION="AuroraGPT"

echo "Eagle project: ${EAGLE_PROJECT}"
echo "Catalyst dir: ${CATALYST_DIR}"
echo "Conda dir: ${CONDA_DIR}"
echo ""

# Step 1: Configure conda to use Eagle (avoid home quota issues)
echo "[1/7] Configuring conda to use Eagle filesystem..."
mkdir -p ${CONDA_DIR}/pkgs
mkdir -p ${CONDA_DIR}/envs

# Put .condarc on Eagle to avoid home quota issues
export CONDARC="${CONDA_DIR}/.condarc"
echo "Configured CONDARC as $CONDARC"

mkdir -p ${CONDA_DIR}
rm -f ~/.condarc  # Remove the file if it exists (might fail, that's ok)
ln -sf ${CONDA_DIR}/.condarc ~/.condarc
touch ${CONDA_DIR}/.condarc

module load conda

# Source conda.sh directly (more reliable than shell hook)
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh

conda config --add pkgs_dirs ${CONDA_DIR}/pkgs
conda config --add envs_dirs ${CONDA_DIR}/envs

echo "  Conda pkgs: ${CONDA_DIR}/pkgs"
echo "  Conda envs: ${CONDA_DIR}/envs"

# Step 2: Create catalyst environment
echo ""
echo "[2/7] Creating catalyst conda environment..."
if conda env list | grep -q "^catalyst "; then
    echo "  Environment 'catalyst' already exists, skipping creation"
else
    conda create -n catalyst python=3.11 -y
fi

# Activate
conda activate catalyst

# Step 3: Install Redis server
echo ""
echo "[3/7] Installing Redis server..."
conda install -c conda-forge redis-server -y

# Step 4: Install Python dependencies
echo ""
echo "[4/7] Installing Python dependencies..."

# Redirect pip cache to Eagle to avoid home quota issues
export PIP_CACHE_DIR="${CONDA_DIR}/pip-cache"
mkdir -p ${PIP_CACHE_DIR}

pip install --quiet redis pyyaml numpy ase
pip install --quiet academy-py
pip install --quiet globus-compute-sdk

# Step 5: Install ML potentials (PyTorch + MACE + CHGNet)
echo ""
echo "[5/7] Installing ML potentials (this may take a few minutes)..."
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
pip install --quiet mace-torch chgnet

# Step 6: Create directories for DFT data
echo ""
echo "[6/7] Setting up DFT data directories..."
mkdir -p ${EAGLE_PROJECT}/pseudopotentials
mkdir -p ${EAGLE_PROJECT}/gpaw-setups
mkdir -p ${CATALYST_DIR}/data

# Step 7: Create activation script
echo ""
echo "[7/7] Creating activation script..."
cat > ${CATALYST_DIR}/activate.sh << 'ACTIVATE'
#!/bin/bash
# Source this to set up Catalyst environment
# Usage: source activate.sh

module load conda
module load cudatoolkit-standalone/12.4.1
module load quantum-espresso/7.3

conda activate catalyst

export EAGLE_PROJECT="/eagle/AuroraGPT/foster"
export PSEUDO_DIR="${EAGLE_PROJECT}/pseudopotentials"
export GPAW_SETUP_PATH="${EAGLE_PROJECT}/gpaw-setups"
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Catalyst environment activated"
echo "  PSEUDO_DIR: ${PSEUDO_DIR}"
echo "  GPAW_SETUP_PATH: ${GPAW_SETUP_PATH}"
ACTIVATE

chmod +x ${CATALYST_DIR}/activate.sh

# Update PBS script with allocation
if [ -f "${CATALYST_DIR}/workflows/polaris/run_agents.pbs" ]; then
    sed -i "s/<YOUR_ALLOCATION>/${ALLOCATION}/g" ${CATALYST_DIR}/workflows/polaris/run_agents.pbs
    echo "  Updated run_agents.pbs with allocation: ${ALLOCATION}"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source ${CATALYST_DIR}/activate.sh"
echo ""
echo "2. Start Redis server:"
echo "   redis-server --port 6379 --bind 0.0.0.0 --daemonize yes"
echo "   redis-cli ping  # Should say PONG"
echo ""
echo "3. Get Argonne inference token:"
echo "   python scripts/inference_auth_token.py"
echo "   export ARGONNE_ACCESS_TOKEN='<token>'"
echo ""
echo "4. Test the setup:"
echo "   python scripts/run_polaris_agents.py --check-env"
echo ""
echo "5. Submit a job:"
echo "   qsub -v REDIS_HOST=polaris.alcf.anl.gov workflows/polaris/run_agents.pbs"
echo ""
echo "6. On your Mac, run:"
echo "   export REDIS_HOST=polaris.alcf.anl.gov"
echo "   python scripts/run_discovery.py --redis-host \$REDIS_HOST"
echo ""
