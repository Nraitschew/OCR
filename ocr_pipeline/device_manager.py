import torch

class VRAMMonitor:
    """Simple VRAM usage monitor using torch.cuda"""
    def get_usage(self):
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated()

def get_total_vram():
    if not torch.cuda.is_available():
        return 0
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).total_memory

class DeviceManager:
    def __init__(self, vram_threshold):
        self.vram_threshold = vram_threshold
        self.gpu_available = torch.cuda.is_available()
        self.vram_monitor = VRAMMonitor()

    def select_optimal_device(self):
        if not self.gpu_available:
            return 'cpu'
        current_vram = self.vram_monitor.get_usage()
        total_vram = get_total_vram()
        if total_vram > 0 and current_vram > self.vram_threshold:
            return 'cpu'
        return 'gpu'
