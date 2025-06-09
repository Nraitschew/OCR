try:
    import torch
except Exception:  # torch may not be installed
    torch = None

class VRAMMonitor:
    """Simple VRAM usage monitor using torch.cuda"""
    def get_usage(self):
        if torch is None or not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated()

def get_total_vram():
    if torch is None or not torch.cuda.is_available():
        return 0
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).total_memory

class DeviceManager:
    def __init__(self, vram_threshold):
        self.vram_threshold = vram_threshold
        self.gpu_available = torch is not None and torch.cuda.is_available()
        self.vram_monitor = VRAMMonitor()

    def select_optimal_device(self):
        if not self.gpu_available:
            return 'cpu'
        try:
            current_vram = self.vram_monitor.get_usage()
            total_vram = get_total_vram()
            if total_vram and current_vram > self.vram_threshold:
                return 'cpu'
            return 'gpu'
        except Exception:
            # on any error fall back to CPU
            return 'cpu'
