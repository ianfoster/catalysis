# M3GNet container with compatible dependencies
FROM python:3.10-slim

WORKDIR /app

# Install build tools for pymatgen compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install matgl with DGL backend
# DGL is required but not auto-installed by matgl
# torchdata<0.8 needed for datapipes compatibility
# numpy<2 for torch 2.1 compatibility
RUN pip install --no-cache-dir \
    "numpy<2" \
    torch==2.1.0 \
    "torchdata<0.8" \
    dgl \
    matgl \
    flask

# Copy the server script
COPY server.py /app/

EXPOSE 8080

CMD ["python", "server.py"]
