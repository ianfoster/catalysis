# Catalyst

Agentic Catalysis Discovery - An autonomous system for catalyst material discovery using LLM-guided optimization and multi-method simulation agents.

## Overview

Catalyst orchestrates closed-loop discovery workflows combining:
- **LLM-based reasoning** for candidate proposal and test selection
- **11 simulation agents** (ML potentials, DFT, MD, kinetics)
- **Distributed computing** via Globus Compute and Academy framework
- **Adaptive budget management** for cost-effective exploration

## Architecture

```
Mac (local)                              Spark (remote)
┌─────────────────┐                      ┌─────────────────────────────────┐
│ GeneratorAgent  │ ───Redis/GC────►     │ vLLM Server (port 8000)         │
│ - propose       │                      │                                 │
│ - collect       │                      │ ShepherdAgent(s)                │
│ - converge      │                      │   └─► LLMProxyAgent             │
└─────────────────┘                      │                                 │
                                         │ Simulation Agents:              │
                                         │   ├─ MACEAgent                  │
                                         │   ├─ CHGNetAgent                │
                                         │   ├─ M3GNetAgent                │
                                         │   ├─ CanteraAgent               │
                                         │   ├─ StabilityAgent             │
                                         │   ├─ SurrogateAgent             │
                                         │   ├─ QEAgent (Quantum ESPRESSO) │
                                         │   ├─ GPAWAgent                  │
                                         │   ├─ OpenMMAgent                │
                                         │   ├─ GROMACSAgent               │
                                         │   └─ CatMAPAgent                │
                                         └─────────────────────────────────┘
```

## Agents

### Orchestration Agents

| Agent | Description |
|-------|-------------|
| **GeneratorAgent** | Proposes catalyst candidates using LLM reasoning. Manages iteration loop and convergence detection. |
| **ShepherdAgent** | Evaluates individual candidates. Uses LLM to select appropriate tests within budget constraints. |
| **LLMProxyAgent** | Centralized LLM access with request tracking, token counting, and latency metrics. |

### Simulation Agents

| Agent | Method | Use Case |
|-------|--------|----------|
| **MACEAgent** | MACE ML potential | Fast screening, near-DFT accuracy |
| **CHGNetAgent** | CHGNet ML potential | Secondary ML screening |
| **M3GNetAgent** | M3GNet ML potential | Alternative ML potential (container-based) |
| **CanteraAgent** | Cantera | Reactor kinetics, CO2 hydrogenation |
| **StabilityAgent** | Thermodynamics | Phase stability analysis |
| **SurrogateAgent** | Physics-informed | Fast surrogate models |
| **QEAgent** | Quantum ESPRESSO | DFT calculations |
| **GPAWAgent** | GPAW | DFT calculations |
| **OpenMMAgent** | OpenMM | Molecular dynamics |
| **GROMACSAgent** | GROMACS | Molecular dynamics |
| **CatMAPAgent** | CatMAP | Microkinetic modeling |

## Quick Start

### Prerequisites

- Python 3.12+
- Redis (for agent coordination)
- Docker (for vLLM on GPU systems)
- Globus Compute credentials

### Installation

```bash
pip install -e .
```

### Running on Mac (Local)

1. **Start Redis** (if not running):
   ```bash
   brew services start redis
   ```

2. **Run discovery** (connects to remote Spark):
   ```bash
   python scripts/run_discovery.py \
       --endpoint $GC_ENDPOINT \
       --llm-url http://<spark-ip>:8000/v1
   ```

### Running on Spark (Remote)

1. **Start vLLM server**:

   See [VLLM_README.md](VLLM_README.md).

2. **Start Academy agents**:
   ```bash
   python scripts/run_spark_agents.py \
       --llm-url http://localhost:8000/v1 \
       --redis-host localhost \
       --redis-port 6379
   ```

   With GPU:
   ```bash
   python scripts/run_spark_agents.py \
       --llm-url http://localhost:8000/v1 \
       --device cuda
   ```

### Using Argonne Inference API

Instead of running a local vLLM server:

```bash
# Get authentication token
python scripts/argonne_auth.py

# Run with Argonne's hosted LLM
python scripts/run_spark_agents.py \
    --llm-url https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1 \
    --llm-model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --redis-host localhost
```

## Configuration

Main configuration in `config.yaml`:

```yaml
run:
  max_iterations: 3
  concurrency: 32

globus_compute:
  endpoint_id: "your-endpoint-id"
  functions:
    fast_surrogate: "function-id"
    ml_screening: "function-id"
    # ... more functions

shepherd:
  llm:
    mode: "shared"
    model: "meta-llama/Meta-Llama-3.1-70B-Instruct"
  budget:
    default: 100.0
    max: 1000.0

academy:
  enabled: true
  llm:
    backend: "vllm"
    vllm:
      model: "meta-llama/Llama-3.1-8B-Instruct"
      port: 8000
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `run_spark_agents.py` | Launch all Academy agents on Spark |
| `run_discovery.py` | Full discovery pipeline orchestration |
| `start_vllm_server.py` | Start vLLM in Docker container |
| `argonne_auth.py` | Get Globus/Argonne auth token |
| `register_gc_functions.py` | Register functions with Globus Compute |
| `monitor_progress.py` | Monitor running discovery |
| `watch_narrative.py` | Real-time discovery narrative |
| `dashboard.py` | Web dashboard for results |
| `agent_status.py` | Check agent health |

## Monitoring

### Check Redis keys
```bash
redis-cli keys '*'
```

### Watch narrative log
```bash
python scripts/watch_narrative.py
```

### Agent status
```bash
python scripts/agent_status.py
```

## Project Structure

```
catalyst/
├── skills/                    # Agent implementations
│   ├── generator.py           # GeneratorAgent
│   ├── shepherd.py            # ShepherdAgent
│   ├── llm_proxy_agent.py     # LLMProxyAgent
│   ├── base_agent.py          # TrackedAgent base class
│   └── sim_agents/            # 11 simulation agents
│       ├── mace_agent.py
│       ├── chgnet_agent.py
│       ├── cantera_agent.py
│       └── ...
├── orchestration/             # Support modules
│   ├── test_registry.py       # Test specifications
│   ├── llm_client.py          # LLM client wrapper
│   ├── generator_prompts.py   # LLM prompts
│   ├── shepherd_prompts.py
│   ├── generator_state.py     # State persistence
│   ├── narrative.py           # Discovery logging
│   └── capabilities.py        # Simulation capabilities
├── hpc/                       # HPC adapters
│   └── globus_compute.py      # Globus Compute wrapper
├── tools/                     # Simulation tool wrappers
│   ├── openmm_tool.py
│   ├── gmx_tool.py
│   └── rdkit_tool.py
├── scripts/                   # Entry points
├── config/                    # Configuration files
├── tests/                     # Test suite
└── data/                      # Results and caches
```

## Discovery Workflow

1. **GeneratorAgent** proposes N candidates using LLM reasoning
2. For each candidate, a **ShepherdAgent** is spawned
3. Shepherd uses LLM to select tests within budget
4. Tests are dispatched to appropriate **SimulationAgents**
5. Results are collected and assessed
6. GeneratorAgent checks for convergence
7. If not converged, iterate with improved proposals

## Development

### Run tests
```bash
pytest tests/
```

### Integration tests (requires services)
```bash
pytest tests/ -m integration
```

## License

See LICENSE file.
