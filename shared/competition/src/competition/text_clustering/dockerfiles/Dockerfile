# Miner solution image: runs untrusted miner code only. No ML stack, no backend
# source, no creds. The ground-truth bake uses Dockerfile.groundtruth instead.
FROM python:3.12-slim

WORKDIR /app

COPY dockerfiles/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the miner-facing python explicitly (NOT `COPY .`) so the backend/common
# source CI stages into the context for the ground-truth build can never leak
# into this image. Keep this in sync with the competition's python files.
COPY __init__.py baseline.py /app/competition/text_clustering/

ENV PYTHONPATH=/app:/workspace:/app/competition/text_clustering

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

ENTRYPOINT ["sh", "-c", "exec \"$@\"", "--"]
CMD ["python", "solution.py"]
