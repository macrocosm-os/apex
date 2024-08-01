poetry install --extras "validator"
poetry run pip uninstall -y uvloop
poetry run python neurons/validator.py --netuid 61 --subtensor.network test --neuron.device cuda --wallet.name validator --wallet.hotkey validator_hotkey --logging.debug --axon.port 22044 --neuron.model_id casperhansen/llama-3-8b-instruct-awq --neuron.llm_max_allowed_memory_in_gb 24 --neuron.gpus 1 --wandb.off