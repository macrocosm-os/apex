FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py .

ARG LLM_MODEL
ENV MODEL_PATH=./downloaded_model

RUN python download_model.py --model-name "$LLM_MODEL" --model-path "$MODEL_PATH"

COPY . .
COPY --from=external_context /vllm_llm.py .

EXPOSE 8000

CMD ["python", "app.py"]
