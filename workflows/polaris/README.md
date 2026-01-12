# Running Catalyst DFT on Polaris

This guide walks through running Academy agents on Polaris to execute DFT calculations (Quantum ESPRESSO, GPAW) for the Catalyst discovery pipeline.

## Architecture

```
Your Mac (local)                         Polaris (ALCF)
┌─────────────────┐                      ┌──────────────────────────────────┐
│ GeneratorAgent  │                      │ PBS Job (run_agents.pbs)         │
│                 │◄─────── Redis ──────►│ ┌──────────────────────────────┐ │
│ run_discovery.py│       Exchange       │ │ Academy Manager              │ │
└─────────────────┘                      │ │                              │ │
                                         │ │  ShepherdAgent(s)            │ │
                                         │ │     │                        │ │
                                         │ │     ├── QEAgent (A100 GPU)   │ │
                                         │ │     ├── GPAWAgent (64 CPUs)  │ │
                                         │ │     ├── MACEAgent (GPU)      │ │
                                         │ │     ├── CHGNetAgent (GPU)    │ │
                                         │ │     └── CanteraAgent         │ │
                                         │ │                              │ │
                                         │ │  LLM: Argonne Inference API  │ │
                                         │ └──────────────────────────────┘ │
                                         └──────────────────────────────────┘
```

The key components:
- **GeneratorAgent** (local): Proposes catalyst candidates
- **ShepherdAgent** (Polaris): Evaluates candidates using simulation agents
- **QEAgent/GPAWAgent** (Polaris): Run expensive DFT calculations
- **Redis Exchange**: Connects local and remote agents

## Prerequisites

1. **ALCF account** with active allocation on Polaris
2. **Redis server** accessible from both your Mac and Polaris
3. **Argonne inference token** for LLM access
4. **Conda environment** on Polaris with dependencies

## Quick Start

### 1. Set up Conda Environment on Polaris

```bash
ssh <username>@polaris.alcf.anl.gov

module load conda
conda create -n catalyst python=3.11
conda activate catalyst

# Install dependencies
pip install academy-agents redis pyyaml
pip install ase numpy torch
pip install mace-torch chgnet  # ML potentials

# Optional: GPAW
pip install gpaw
```

### 2. Get Argonne Inference Token

```bash
# On Polaris (or locally)
python scripts/inference_auth_token.py

# This opens a browser for Globus auth
# Copy the token to your environment:
export ARGONNE_ACCESS_TOKEN="<token>"

# Add to ~/.bashrc for persistence
echo 'export ARGONNE_ACCESS_TOKEN="<token>"' >> ~/.bashrc
```

### 3. Set up Redis Exchange

You need a Redis server accessible from both your Mac and Polaris. Options:

**Option A: Use existing Redis server**
```bash
# If your institution has a Redis server, use that
export REDIS_HOST=redis.example.com
```

**Option B: Run Redis on a cloud VM**
```bash
# On a VM with public IP
docker run -d -p 6379:6379 redis:latest
# Use the VM's IP as REDIS_HOST
```

**Option C: SSH tunnel (for testing)**
```bash
# On your Mac, tunnel Redis through Polaris login node
ssh -L 6379:localhost:6379 <user>@polaris.alcf.anl.gov

# Then on Polaris, start Redis
redis-server --port 6379

# Use localhost as REDIS_HOST on both ends
```

### 4. Submit PBS Job on Polaris

```bash
cd /path/to/catalyst/workflows/polaris

# Edit the PBS script to set your allocation
vi run_agents.pbs
# Change: #PBS -A <YOUR_ALLOCATION>

# Submit with Redis host
qsub -v REDIS_HOST=<redis-server> run_agents.pbs

# Check job status
qstat -u $USER
```

### 5. Run GeneratorAgent Locally

On your Mac:

```bash
# Set Redis connection
export REDIS_HOST=<redis-server>
export REDIS_PORT=6379

# Run discovery with Polaris agents
python scripts/run_discovery.py --redis-host $REDIS_HOST
```

## PBS Job Options

Edit `run_agents.pbs` for your needs:

| Setting | Default | Description |
|---------|---------|-------------|
| `select=1:ncpus=64:ngpus=4` | 1 node | Compute resources |
| `walltime=01:00:00` | 1 hour | Job duration |
| `queue=debug` | debug | PBS queue |
| `-A` | - | Your allocation (required) |

### Queue Options

| Queue | Max Nodes | Max Time | Use Case |
|-------|-----------|----------|----------|
| debug | 10 | 1 hour | Testing |
| debug-scaling | 64 | 1 hour | Scaling tests |
| prod | 560 | 24 hours | Production |
| preemptable | 10 | 72 hours | Low priority |

## Agent Configuration

Customize agents in `run_polaris_agents.py` or via command line:

```bash
# DFT only (QE + GPAW)
python scripts/run_polaris_agents.py \
    --agents qe gpaw \
    --redis-host $REDIS_HOST

# Full suite with more shepherds
python scripts/run_polaris_agents.py \
    --agents qe gpaw mace chgnet cantera stability surrogate \
    --num-shepherds 4 \
    --budget 300 \
    --redis-host $REDIS_HOST

# Check environment without running
python scripts/run_polaris_agents.py --check-env
```

## Pseudopotentials Setup

QE needs pseudopotentials:

```bash
# Create directory on Eagle filesystem
mkdir -p /eagle/projects/<YOUR_PROJECT>/pseudopotentials
cd /eagle/projects/<YOUR_PROJECT>/pseudopotentials

# Download SSSP efficiency pseudopotentials
# From: https://www.materialscloud.org/discover/sssp/table/efficiency

# Set in environment
export PSEUDO_DIR=/eagle/projects/<YOUR_PROJECT>/pseudopotentials
```

## Troubleshooting

### PBS job fails to start
```bash
# Check job output
cat polaris_agents.log

# Common issues:
# - Wrong allocation name
# - Missing modules
# - Conda env not found
```

### Agents can't connect to Redis
```bash
# Test Redis connection from Polaris
python -c "import redis; r=redis.Redis('$REDIS_HOST'); print(r.ping())"

# Check firewall rules
# Redis port 6379 must be open
```

### QE not found
```bash
# Load the module
module load quantum-espresso/7.3
which pw.x
```

### GPAW import fails
```bash
# Check conda environment
conda activate catalyst
python -c "import gpaw; print(gpaw.__version__)"

# If missing, install
pip install gpaw
```

### LLM calls timeout
```bash
# Check token is valid
python scripts/inference_auth_token.py --check

# Regenerate if expired
python scripts/inference_auth_token.py
```

### Agents don't receive tasks
```bash
# Check Redis exchange
python -c "
import redis
r = redis.Redis('$REDIS_HOST')
print('Keys:', r.keys('*'))
"

# Verify GeneratorAgent is using same Redis
```

## Files Reference

| File | Purpose |
|------|---------|
| `run_agents.pbs` | PBS job script to launch agents |
| `../run_polaris_agents.py` | Agent launcher script |
| `setup_endpoint.py` | (Alternative) GC endpoint setup |
| `dft_functions.py` | (Alternative) GC function definitions |

## Example Workflow

1. **Start Polaris agents** (on Polaris):
   ```bash
   qsub -v REDIS_HOST=redis.example.com workflows/polaris/run_agents.pbs
   ```

2. **Monitor job**:
   ```bash
   qstat -u $USER
   tail -f polaris_agents.log
   ```

3. **Run discovery** (on Mac):
   ```bash
   python scripts/run_discovery.py \
       --redis-host redis.example.com \
       --max-iterations 10
   ```

4. **Check results**:
   ```bash
   # View discovered catalysts
   cat data/generator_results.jsonl | jq '.score'
   ```

## Cost Considerations

DFT calculations on Polaris consume allocation hours:

| Calculation | Approx. Time | Node-hours |
|-------------|--------------|------------|
| QE SCF (small) | 5-10 min | 0.08-0.17 |
| QE relax | 30-60 min | 0.5-1.0 |
| GPAW SCF | 10-20 min | 0.17-0.33 |
| ML screening | 10-30 sec | ~0.01 |

Use `--budget` to limit DFT calls per candidate, and test with `debug` queue first.
