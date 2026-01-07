# Catalyst

Agentic Catalysis Discovery - An autonomous system for catalyst material discovery using LLM-guided optimization and multi-method simulation agents.

## Overview

Catalyst orchestrates closed-loop discovery workflows combining:
- **LLM-based reasoning** for candidate proposal and test selection
- **Simulation agents** (ML potentials, DFT, kinetics)
- **Distributed computing** via Globus Compute and Academy framework
- **Adaptive budget management** for cost-effective exploration

## Architecture

```
Mac (local)                              Spark (remote)
┌─────────────────┐                      ┌─────────────────────────────────┐
│ GeneratorAgent  │◄──SSH tunnel────────►│ Redis (port 6379)               │
│ (OpenAI LLM)    │   (port 6380)        │                                 │
│                 │                      │ vLLM Server (port 8000)         │
│ run_catalyst.py │◄──Globus Compute────►│   Llama-3.1-8B-Instruct         │
└─────────────────┘                      │                                 │
                                         │ ShepherdAgents (x4)             │
                                         │   └─► LLMProxyAgent             │
                                         │                                 │
                                         │ Simulation Agents:              │
                                         │   ├─ MACEAgent      [ML]        │
                                         │   ├─ CHGNetAgent    [ML]        │
                                         │   ├─ CanteraAgent   [kinetics]  │
                                         │   ├─ StabilityAgent [thermo]    │
                                         │   └─ SurrogateAgent [fast]      │
                                         └─────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.12+
- Redis running on Spark
- Docker (for vLLM on GPU systems)
- OpenAI API key (for Generator LLM)
- Globus Compute endpoint on Spark

### Installation

```bash
pip install -e .
```

### One-Command Workflow

The unified `run_catalyst.py` script handles everything:

```bash
# Set your API key and endpoint
export OPENAI_API_KEY="sk-..."
export GC_ENDPOINT="your-endpoint-id"

# Full workflow: start agents, create tunnel, run discovery
python scripts/run_catalyst.py \
    --endpoint $GC_ENDPOINT \
    --spark-host spark \
    --device cpu
```

This will:
1. Start vLLM server on Spark (if not running)
2. Start ShepherdAgents and SimulationAgents on Spark
3. Create SSH tunnel to Spark's Redis
4. Run GeneratorAgent locally with OpenAI
5. Coordinate discovery via Redis

### Common Operations

```bash
# Check agent status
python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --check-agents

# Restart agents (stops, clears cache, updates code, starts)
python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --restart-agents --start-agents-only --device cpu

# Run discovery (agents already running)
python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --spark-host spark --skip-agents

# Clear all caches
python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --clear-cache

# View agent logs
python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --logs

# Update code on Spark
python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --update-code
```

### Manual Workflow (Alternative)

If you prefer manual control:

**On Spark:**
```bash
# Start vLLM server
python scripts/start_vllm_server.py --model meta-llama/Llama-3.1-8B-Instruct

# Start agents
python scripts/run_spark_agents.py \
    --llm-url http://localhost:8000/v1 \
    --redis-host localhost \
    --num-shepherds 4 \
    --device cpu
```

**On Mac:**
```bash
# Create SSH tunnel
ssh -N -L 6380:localhost:6379 spark &

# Run generator
python scripts/run_generator.py \
    --redis-host localhost \
    --redis-port 6380 \
    --generator-llm openai
```

## Agents

### Orchestration Agents

| Agent | Location | Description |
|-------|----------|-------------|
| **GeneratorAgent** | Mac | Proposes candidates using OpenAI. Manages iteration loop and convergence. |
| **ShepherdAgent** | Spark | Evaluates candidates. Uses local vLLM to select tests within budget. |
| **LLMProxyAgent** | Spark | Centralized LLM access with request tracking and metrics. |

### Simulation Agents

| Agent | Method | Status | Use Case |
|-------|--------|--------|----------|
| **MACEAgent** | MACE ML potential | ✅ Active | Fast screening, structure relaxation |
| **CHGNetAgent** | CHGNet ML potential | ✅ Active | Alternative ML screening |
| **CanteraAgent** | Cantera | ✅ Active | Reactor kinetics, CO2 hydrogenation |
| **StabilityAgent** | Thermodynamics | ✅ Active | Phase stability analysis |
| **SurrogateAgent** | Physics-informed | ✅ Active | Fast surrogate models |
| **M3GNetAgent** | M3GNet ML potential | ⚠️ Container | Requires separate container |
| **QEAgent** | Quantum ESPRESSO | ❌ Disabled | DFT (not configured) |
| **GPAWAgent** | GPAW | ❌ Disabled | DFT (not configured) |
| **OpenMMAgent** | OpenMM | ❌ Disabled | No force field for inorganics |
| **GROMACSAgent** | GROMACS | ❌ Disabled | Not implemented |
| **CatMAPAgent** | CatMAP | ❌ Disabled | Not implemented |

## Configuration

Main configuration in `config.yaml`:

```yaml
run:
  max_iterations: 3
  concurrency: 32

academy:
  enabled: true
  llm:
    backend: "vllm"
    vllm:
      model: "meta-llama/Llama-3.1-8B-Instruct"
      port: 8000

  shepherds:
    num_concurrent: 4
    budget_per_candidate: 150.0
    timeout: 3600
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| **`run_catalyst.py`** | **Unified launcher** - handles agents, tunnel, and discovery |
| `run_spark_agents.py` | Launch Academy agents on Spark (manual mode) |
| `run_generator.py` | Run GeneratorAgent standalone (manual mode) |
| `start_vllm_server.py` | Start vLLM in Docker container |
| `argonne_auth.py` | Get Globus/Argonne auth token |
| `watch_narrative.py` | Real-time discovery narrative via Redis |
| `dashboard.py` | Status dashboard |

### run_catalyst.py Options

```
Connection:
  --endpoint ID         Globus Compute endpoint ID on Spark
  --spark-host HOST     Spark hostname for SSH (default: spark)
  --redis-port PORT     Local Redis port for tunnel (default: 6380)

Agent Management:
  --check-agents        Check agent status and exit
  --stop-agents         Stop agents and exit
  --restart-agents      Stop, clear cache, update code, restart
  --start-agents-only   Start agents and exit (no discovery)
  --update-code         Git pull on Spark
  --clear-cache         Clear local and Redis caches
  --logs                Show agent logs from Spark

Skip Options:
  --skip-agents         Don't start agents (already running)
  --skip-tunnel         Don't create SSH tunnel (already open)

Agent Config:
  --llm-model MODEL     LLM model on Spark
  --num-shepherds N     Number of ShepherdAgents (default: 4)
  --device {cpu,cuda}   Device for ML agents

Generator Config:
  --generator-llm TYPE  openai or argonne (default: openai)
  --max-iterations N    Max discovery iterations (default: 3)
  --candidates-per-iteration N  Candidates per iteration (default: 6)
  --budget FLOAT        Budget per candidate (default: 100.0)
```

## Monitoring

### Watch discovery narrative
```bash
python scripts/watch_narrative.py --redis-host localhost --redis-port 6380
```

### Check Redis keys
```bash
ssh spark 'redis-cli keys "*"'
```

### View agent logs
```bash
python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --logs
```

## Project Structure

```
catalyst/
├── skills/                    # Agent implementations
│   ├── generator.py           # GeneratorAgent
│   ├── shepherd.py            # ShepherdAgent
│   ├── llm_proxy_agent.py     # LLMProxyAgent
│   ├── base_agent.py          # TrackedAgent base class
│   └── sim_agents/            # Simulation agents
│       ├── mace_agent.py
│       ├── chgnet_agent.py
│       ├── cantera_agent.py
│       └── ...
├── orchestration/             # Support modules
│   ├── test_registry.py       # Test specifications & runtime tracking
│   ├── llm_client.py          # LLM client wrapper
│   ├── generator_prompts.py   # Generator LLM prompts
│   ├── shepherd_prompts.py    # Shepherd LLM prompts
│   ├── generator_state.py     # State persistence
│   ├── narrative.py           # Discovery logging to Redis
│   └── capabilities.py        # Simulation capabilities
├── hpc/                       # HPC adapters
│   └── globus_compute.py      # Globus Compute wrapper
├── scripts/                   # Entry points
├── config/                    # Configuration files
├── tests/                     # Test suite
└── data/                      # Results and caches
```

## Discovery Workflow

1. **GeneratorAgent** (Mac) proposes N candidates using OpenAI
2. Candidates sent to **ShepherdAgents** (Spark) via Redis
3. Each Shepherd runs two phases:
   - **Phase 1**: All fast tests in parallel (ML screening, surrogate)
   - **Phase 2**: LLM selects slow tests based on Phase 1 results
4. Tests dispatched to appropriate **SimulationAgents**
5. Results assessed by Shepherd's LLM → viability score
6. Scores returned to Generator
7. Generator checks convergence; if not converged, proposes new candidates

## Development

### Run tests
```bash
pytest tests/
```

### Tag releases
```bash
git tag -a v0.1-description -m "Description"
git push origin v0.1-description
```

## License

See LICENSE file.
