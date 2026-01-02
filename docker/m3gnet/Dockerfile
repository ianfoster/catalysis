# M3GNet container with compatible dependencies
FROM python:3.10-slim

WORKDIR /app

# Install dependencies with compatible versions
# matgl 1.1.3 requires dgl>=2.0.0
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    dgl==2.0.0 \
    matgl==1.1.3 \
    ase==3.22.1 \
    pymatgen==2024.2.8 \
    numpy \
    flask

# Copy the server script
COPY server.py /app/

EXPOSE 8080

CMD ["python", "server.py"]
