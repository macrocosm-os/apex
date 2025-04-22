from prompting.llms.vllm_llm import ReproducibleVLLM


class VLLMTextImageToText(ReproducibleVLLM):
    # def __init__(
    #     self,
    #     model_id: str = "google/gemma-3-27b-it",
    #     device: str = "cuda:0",
    #     sampling_params: dict[str, str | float | int | bool] | None = None,
    # ):
    #     super().__init__(model_id, device, sampling_params)
    # self.model: AutoModelForImageTextToText = AutoModelForImageTextToText.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map=self._device,
    # )
    # self.tokenizer = AutoProcessor.from_pretrained(model_id)
    # self.valid_generation_params = set(self.model.generation_config.to_dict().keys())
    # self.message_formatter = HFTextImageToText.format_messages

    @staticmethod
    def format_messages(messages: list[str] | list[dict[str, str]]) -> list[dict[str, str | list[dict[str, str]]]]:
        """Format the messages for the gemma model.

        Converts message content strings to dictionaries with type and text fields.
        Example:
        Input: [{"role": "user", "content": "Hello"}]
        Output: [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        """
        formatted_messages = []
        # Flatten list if messages is nested
        if isinstance(messages, list) and all(isinstance(m, list) for m in messages):
            messages = [item for sublist in messages for item in sublist]
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                # If content is a string, convert it to a list with a dictionary
                if isinstance(message["content"], str):
                    formatted_message = message.copy()
                    formatted_message["content"] = [{"type": "text", "text": message["content"]}]
                    formatted_messages.append(formatted_message)
                else:
                    # If content is already in the correct format, keep it as is
                    formatted_messages.append(message)
            else:
                # Handle other message formats if needed
                formatted_messages.append(message)

        return formatted_messages


if __name__ == "__main__":
    model = VLLMTextImageToText(model_id="google/gemma-3-27b-it", device="cuda:0")
    print(model.generate([{"role": "user", "content": "What's ur name?"}]))
