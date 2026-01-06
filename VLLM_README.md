# catalysis
Agentic Catalysis

## Starting the vLLM Server

### Prerequisites

- NVIDIA GPU with sufficient memory (GB10 or similar)
- Docker with NVIDIA Container Toolkit
- Local model cache at `~/.cache/huggingface/` (or HuggingFace token for gated models)

### Quick Start

**Option 1: Using the startup script (Recommended)**

```bash
python scripts/start_vllm_server.py --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8
```

**Option 2: Direct Docker command**

```bash
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -d --name vllm-server \
  nvcr.io/nvidia/vllm:25.11-py3 \
  vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager \
  --trust-remote-code \
  --max-model-len 120000
```

### Important Configuration Notes

1. **--enforce-eager**: Required for GB10 GPUs
2. **--max-model-len 120000**: Reduced from default 131K to fit KV cache in GPU memory
3. **Volume mount**: `-v ~/.cache/huggingface:/root/.cache/huggingface` uses local model cache

### Model Loading Time

- Llama 3.1 70B FP8: ~6-7 minutes
- Llama 3.1 8B: ~1-2 minutes

### Recommended Models

Models that work well on GB10 with --enforce-eager:

- `neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8` (70B, 67.7 GiB, best quality)
- `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` (8B, faster)
- `Qwen/Qwen3-14B-FP8` (14B, good balance)
- `Qwen/Qwen3-30B-A3B-FP8` (30B MoE, good speed)

### Testing the Server

**Check health:**
```bash
curl http://localhost:8000/health
```

**List available models:**
```bash
curl http://localhost:8000/v1/models
```

**Test chat completion:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Monitoring the Server

**View logs:**
```bash
docker logs vllm-server
```

**Follow logs in real-time:**
```bash
docker logs -f vllm-server
```

**Check GPU usage:**
```bash
nvidia-smi
```

### Stopping the Server

```bash
docker stop vllm-server
docker rm vllm-server
```

### Troubleshooting

**Issue: "401 Client Error: Unauthorized" or "GatedRepoError"**

Solution: Model requires HuggingFace authentication. Either:
1. Use the FP8 variant (e.g., `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic`)
2. Add HuggingFace token: `-e HF_TOKEN=your_token_here`

**Issue: "ValueError: KV cache memory needed ... larger than available"**

Solution: Reduce max context length with `--max-model-len`:
```bash
# For 70B model on GB10, use 120000 instead of default 131072
--max-model-len 120000
```

**Issue: Container exits immediately**

Solution: Check logs with `docker logs vllm-server` for specific error

**Issue: Model downloading instead of using cache**

Solution: Verify volume mount is correct:
```bash
ls ~/.cache/huggingface/hub/models--neuralmagic--Meta-Llama-3.1-70B-Instruct-FP8
```

### Advanced Usage

**Using the Python script with custom settings:**

```bash
# Start with custom context length
python scripts/start_vllm_server.py \
  --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --max-model-len 100000 \
  --port 8000

# Start via Globus Compute (remote execution)
python scripts/start_vllm_server.py \
  --endpoint $GC_ENDPOINT \
  --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8

# With custom startup timeout
python scripts/start_vllm_server.py \
  --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --startup-timeout 600
```

**Script options:**
- `--model`: HuggingFace model name (required)
- `--port`: Server port (default: 8000)
- `--max-model-len`: Max context length (auto-calculated if not specified)
- `--no-enforce-eager`: Disable eager mode (not recommended for GB10)
- `--startup-timeout`: Model loading timeout in seconds (default: 300)
- `--output`: Output JSON file path (default: vllm_server_info.json)
- `--endpoint`: Globus Compute endpoint for remote execution

The script automatically:
- Stops existing vllm-server containers
- Mounts your HuggingFace cache directory
- Waits for model loading and server readiness
- Tests the server with a sample request
- Saves connection info to `vllm_server_info.json`
