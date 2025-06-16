import re

import numpy as np
import torch
from loguru import logger

from prompting.llms.hf_llm import ReproducibleHF
from prompting.llms.hf_text_image import VLLMTextImageToText
from prompting.llms.vllm_llm import ReproducibleVLLM
from shared.misc import classproperty


def contains_gpu_index_in_device(device: str) -> bool:
    pattern = r"^cuda:\d+$"
    return bool(re.match(pattern, device))


def calculate_single_gpu_requirements(device: str, max_allowed_memory_allocation_in_bytes: int):
    if contains_gpu_index_in_device(device):
        device_with_gpu_index = device
    else:
        device_with_gpu_index = torch.cuda.current_device()

    torch.cuda.synchronize()
    global_free, total_gpu_memory = torch.cuda.mem_get_info(device=device_with_gpu_index)

    logger.info(f"Available free memory: {round(global_free / 10e8, 2)} GB")
    logger.info(f"Total gpu memory {round(total_gpu_memory / 10e8, 2)} GB")

    if global_free < max_allowed_memory_allocation_in_bytes:
        ex = Exception(
            f"Not enough memory to allocate for the model. Please ensure you have at least {max_allowed_memory_allocation_in_bytes / 10e8} GB of free GPU memory."
        )
        logger.error(ex)
        raise ex

    gpu_utilization = round(max_allowed_memory_allocation_in_bytes / global_free, 2)
    logger.info(
        f'{gpu_utilization * 100}% of the GPU memory will be utilized for loading the model to device "{device}".'
    )

    return gpu_utilization


def calculate_multiple_gpu_requirements(device: str, gpus: int, max_allowed_memory_allocation_in_bytes: int):
    torch.cuda.synchronize()
    total_free_memory = 0
    total_gpu_memory = 0

    for i in range(gpus):
        gpu_device = f"cuda:{i}"
        global_free, total_memory = torch.cuda.mem_get_info(device=gpu_device)
        total_free_memory += global_free
        total_gpu_memory += total_memory

    logger.info(f"Total available free memory across all visible {gpus} GPUs: {round(total_free_memory / 10e8, 2)} GB")
    logger.info(f"Total GPU memory across all visible GPUs: {gpus} {round(total_gpu_memory / 10e8, 2)} GB")

    if total_free_memory < max_allowed_memory_allocation_in_bytes:
        raise torch.cuda.CudaError(
            f"Not enough memory across all specified {gpus} GPUs to allocate for the model. Please ensure you have at least {max_allowed_memory_allocation_in_bytes / 10e8} GB of free GPU memory."
        )

    gpu_utilization = round(max_allowed_memory_allocation_in_bytes / total_free_memory, 2)
    logger.info(
        f"{gpu_utilization * 100}% of the total GPU memory across all GPUs will be utilized for loading the model."
    )

    return gpu_utilization


def calculate_gpu_requirements(
    device: str,
    gpus: int,
    max_allowed_memory_allocation_in_bytes: float,
) -> float:
    """Calculates the memory utilization requirements for the model to be loaded on the device.
    Args:
        device (str): The device to load the model to.
        max_allowed_memory_allocation_in_bytes (int, optional): The maximum allowed memory allocation in bytes. Defaults to 20e9 (20GB).
    """
    if gpus == 1:
        return calculate_single_gpu_requirements(device, max_allowed_memory_allocation_in_bytes)
    else:
        return calculate_multiple_gpu_requirements(
            device, gpus, max_allowed_memory_allocation_in_bytes=max_allowed_memory_allocation_in_bytes
        )


class GPUInfo:
    def log_gpu_info():
        logger.info(
            f"""Total GPU memory: {GPUInfo.total_memory} GB
                    Free GPU memory: {GPUInfo.free_memory} GB
                    Used GPU memory: {GPUInfo.used_memory} GB
                    GPU utilization: {GPUInfo.gpu_utilization * 100}%"""
        )

    @classproperty
    def total_memory(cls):
        return np.sum([torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(cls.n_gpus)])

    @classproperty
    def used_memory(cls):
        return cls.total_memory - cls.free_memory

    @classproperty
    def free_memory(cls):
        return np.sum([torch.cuda.mem_get_info(i)[0] / (1024**3) for i in range(cls.n_gpus)])

    @classproperty
    def n_gpus(cls):
        return torch.cuda.device_count()

    @classproperty
    def gpu_utilization(cls):
        return cls.used_memory / cls.total_memory


TEXT_MODELS: set[str | None] = set([None, "mrfakename/mistral-small-3.1-24b-instruct-2503-hf"])


def model_factory(model_name: str) -> type[ReproducibleHF]:
    if model_name in TEXT_MODELS:
        return ReproducibleVLLM
    else:
        return VLLMTextImageToText
