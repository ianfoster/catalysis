# M3GNet container with compatible dependencies
FROM python:3.10-slim

WORKDIR /app

# Install build tools for pymatgen compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install numpy first, then torch 2.0 for better ARM support
RUN pip install --no-cache-dir "numpy<2"
RUN pip install --no-cache-dir torch==2.0.0

# Install matgl with DGL backend
# DGL 1.1.3 is compatible with torch 2.0
# torchdata<0.8 needed for datapipes compatibility
RUN pip install --no-cache-dir \
    "torchdata<0.8" \
    dgl==1.1.3 \
    matgl \
    flask

# Copy the server script
COPY server.py /app/

# Disable oneDNN for ARM compatibility
ENV DNNL_VERBOSE=0
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TORCH_USE_RTLD_GLOBAL=1

EXPOSE 8080

CMD ["python", "server.py"]
