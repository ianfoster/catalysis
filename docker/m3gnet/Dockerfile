# M3GNet container with compatible dependencies
FROM python:3.10-slim

WORKDIR /app

# Install build tools for pymatgen compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install matgl and let it pull compatible dependencies
# Then add flask for HTTP server
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    matgl \
    flask

# Copy the server script
COPY server.py /app/

EXPOSE 8080

CMD ["python", "server.py"]
