# Miner Documentation

## Overview

The Apex Miner is a minimalistic implementation designed for educational purposes within the Bittensor network. It provides both **generator** and **discriminator** functionality for the Apex subnet.

> ⚠️ **WARNING**: This is a dummy miner implementation. Do not run it on mainnet as it won't produce any yields and the registration fee will be lost.

## Architecture

### Core Components

1. **HTTP Server**: Uses aiohttp to serve requests on `/v1/chat/completions` endpoint
2. **Bittensor Integration**: Connects to Bittensor network using subtensor and wallet
3. **Dual Functionality**: Acts as both generator and discriminator

### Network Configuration

- **Testnet**: NetUID 61
- **Mainnet**: NetUID 1 (not recommended for this dummy implementation)

## Functionality

### Generator Mode

When `step` parameter equals `"generator"`:
- Receives a query in the request body
- Returns a generation response
- The miner should try to match the validator's reference as close as possible, within a much shorter time.

```python
# Current dummy implementation
response = "This is a dummy generation from the base miner"
```

### Discriminator Mode

When `step` parameter is not `"generator"`:
- Receives query and generation in request body
- Should classify the response (Miner class: 1, Validator class: 0)

```python
# Current dummy implementation
response = random.choice(["0", "1"])
```

## Setup and Usage

### Prerequisites

- Python environment with required dependencies
- Bittensor wallet (coldkey and hotkey)
   - To register a neuron, see [the documentation](https://docs.learnbittensor.org/miners/#miner-registration).
- Network access for IP detection and subnet communication

### Command Line Arguments

```bash
python miner.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--network` | "test" | Network type: "test" or "finney" |
| `--coldkey` | Required | Coldkey name for wallet |
| `--hotkey` | Required | Hotkey name for wallet |
| `--port` | 8080 | Port to serve the miner on |

### Example Usage

```bash
# Run on testnet
python miner.py --network test --coldkey my_coldkey --hotkey my_hotkey --port 8080

# Run on finney (not recommended for dummy miner)
python miner.py --network finney --coldkey my_coldkey --hotkey my_hotkey --port 8080
```

## Request Format

The miner accepts POST requests to `/v1/chat/completions` with JSON body:

### Generator Request
```json
{
  "step": "generator",
  "query": "Your input query here"
}
```

### Discriminator Request
```json
{
  "step": "discriminator",
  "query": "Input query",
  "generation": "Response to classify"
}
```

## Miner Ecosystem

### Miner Sampling

The validator system includes a `MinerSampler` class that:
- Samples miners from the network metagraph
- Supports random and sequential sampling modes
- Queries multiple miners simultaneously
- Tracks miner information (hotkey, UID, address)

### Scoring System

The `MinerScorer` evaluates miner performance:
- Uses a 22-hour scoring window
- Sets weights based on performance
- Updates scores periodically (every 22 hours by default)
- Connects to results database for historical data

## Development TODOs

### Critical Implementation Needed

1. **Generator Function**: Replace dummy response with actual text generation
   - Implement proper language model or API integration
   - Handle different types of queries appropriately

2. **Discriminator Function**: Implement actual classification logic
   - Develop criteria for miner vs validator classification
   - Use query and generation context for accurate scoring

3. **Error Handling**: Improve robustness
   - Better request validation
   - Network error handling
   - Graceful degradation strategies

## Security Considerations

- **Mainnet Warning**: This implementation is not production-ready
- **Registration Fees**: Running on mainnet will result in lost registration fees
- **IP Exposure**: Miner IP is publicly registered on the network
- **Wallet Security**: Protect coldkey and hotkey credentials

## Integration Points

### Validator Integration
- Validators query miners using the `/v1/chat/completions` endpoint
- Results are used for miner scoring and weight setting
- Performance data is logged to database

### Network Integration
- Automatic IP detection using AWS checkip service
- Subnet registration via Bittensor serve_extrinsic
- Protocol version 4 compliance

## Troubleshooting

### Common Issues

1. **Port Already in Use**: Change the `--port` parameter
2. **Wallet Not Found**: Verify coldkey and hotkey names
3. **Network Connection**: Check internet connectivity and firewall settings
4. **Subnet Registration**: Ensure sufficient balance for registration fees