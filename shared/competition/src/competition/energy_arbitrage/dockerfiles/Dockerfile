FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY dockerfiles/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python competition code to the nested path that test_challenge.py's
# __file__-based import root computation expects.
COPY python/ /app/competition/energy_arbitrage/python/
COPY shared/backend/ /app/shared/backend/
COPY shared/common/ /app/shared/common/

ENV PYTHONPATH=/app:/workspace:/app/shared/backend/src:/app/shared/common/src

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

ENTRYPOINT ["sh", "-c", "exec \"$@\"", "--"]
CMD ["python", "/app/competition/energy_arbitrage/python/test_challenge.py"]
