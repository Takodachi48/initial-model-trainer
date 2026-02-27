import torch
from typing import Union


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        else:
            print(f"Using CUDA device: {device}")
    
    device_obj = torch.device(device)
    
    # Print device info
    if device_obj.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device_obj)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(device_obj).total_memory / 1e9:.1f} GB")
    
    return device_obj


def set_device(device: Union[str, torch.device]) -> torch.device:
    """
    Set and return the specified device.
    
    Args:
        device: Device specification
        
    Returns:
        PyTorch device
    """
    if isinstance(device, str):
        device_obj = get_device(device)
    else:
        device_obj = device
    
    return device_obj


def get_device_info(device: torch.device) -> dict:
    """
    Get detailed information about the device.
    
    Args:
        device: PyTorch device
        
    Returns:
        Dictionary with device information
    """
    info = {
        'device_type': device.type,
        'device_index': device.index if device.type == 'cuda' else None
    }
    
    if device.type == "cuda":
        info.update({
            'device_name': torch.cuda.get_device_name(device),
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'memory_allocated': torch.cuda.memory_allocated(device) / 1e9,
            'memory_reserved': torch.cuda.memory_reserved(device) / 1e9,
            'max_memory': torch.cuda.get_device_properties(device).total_memory / 1e9
        })
    
    return info


def clear_gpu_cache():
    """Clear GPU cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
