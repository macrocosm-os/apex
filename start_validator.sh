cp .env src/validator/.env
cd src/validator
uv sync
uv run --package validator main.py
