# M3GNet container with compatible dependencies
FROM python:3.10-slim

WORKDIR /app

# Install build tools for pymatgen compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies with compatible versions
# matgl 1.1.3 requires dgl>=2.0.0 and newer ASE with ase.filters
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    dgl==2.0.0 \
    matgl==1.1.3 \
    ase>=3.23.0 \
    pymatgen==2024.2.8 \
    numpy \
    flask

# Copy the server script
COPY server.py /app/

EXPOSE 8080

CMD ["python", "server.py"]
