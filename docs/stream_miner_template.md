# Creating Stream Miners

Miners for SN1 **must** support the StreamPromptingSynapse. This enables all miners on the network to stream batches of tokens to the validator. This has clear benefits, such as:

1. Getting rewards for partial responses, and
2. Enabling better user-product interactivity when using a frontend.

Getting custom miners to use streaming is a large engineering effort. To make this effort easier, we have provided a simple template below that emphasizes the important components needed for streaming.

## Architecture

Miner architectures require that you are running a synchronous `forward` method, with an internal `async _forward` function. The code below provides a basic outline of how the `async _forward` function should be structured. There are two main points here:

1. Adding data to the buffer and sending it when it reaches the `config.neuron.streaming_batch_size`
2. Sending the final buffer of data if inference is finished, and there are fewer tokens than the batch size.

```python
def forward(self, synapse: StreamPromptingSynapse) -> Awaitable:
    async def _forward(
        self,
        **kwargs,
        streamer,
        send: Send,
    ):

        buffer = []
        timeout_reached = False

        try:
            for token in streamer:
                buffer.append(token)

                if time.time() - init_time > timeout_threshold:
                    bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
                    timeout_reached = True
                    break

                if len(buffer) == self.config.neuron.streaming_batch_size:
                    joined_buffer = "".join(buffer)
                    bt.logging.debug(f"Streamed tokens: {joined_buffer}")

                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    buffer = []

            if (
                buffer and not timeout_reached
            ):  # Don't send the last buffer of data if timeout.
                joined_buffer = "".join(buffer)
                temp_completion += joined_buffer
                bt.logging.debug(f"Streamed tokens: {joined_buffer}")

                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": False,
                    }
                )

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            if self.config.neuron.stop_on_forward_exception:
                self.should_exit = True

    token_streamer = partial(
        _forward,
        self,
        **kwargs,
        streamer
    )

    return synapse.create_streaming_response(token_streamer)
```

HuggingFace miners require you to run a separate inference thread in the background, add to a queue, and manually clear it at the end of the `async _forward` method.

This branch contains multiple implementations. To see:
1. Langchain+OpenAI implementation, refer to `prompting/miners/openai_miner.py`
2. HuggingFace implementation, refer to `prompting/miners/hf_miner.py`

It is **necessary** that the forward method of the miner class returns this `synapse.create_streaming_response(token_streamer)`. As seen, the `token_streamer` is a partial function that takes in a `send` packet. This packet will be sent by the bittensor middleware to facilitate the communications between the validator and the miner. You do **not** need to modify any logic around the `send` packet, as this is the same for **all** miners.
