"""
Resource manager for controlling CPU and memory usage
"""
import os
import psutil
import resource
import threading
import time
from typing import Optional
import logging
from .config import MAX_MEMORY_GB, TARGET_MEMORY_PERCENT, TARGET_CPU_PERCENT

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages system resources to stay within configured limits"""
    
    def __init__(self):
        self.max_memory_bytes = int(MAX_MEMORY_GB * 1024 * 1024 * 1024)
        self.target_memory_percent = TARGET_MEMORY_PERCENT
        self.target_cpu_percent = TARGET_CPU_PERCENT
        self.monitoring = False
        self.monitor_thread = None
        self._process = psutil.Process(os.getpid())
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitor_thread.start()
            logger.info("Resource monitoring started")
            
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            logger.info("Resource monitoring stopped")
            
    def _monitor_resources(self):
        """Monitor resources and log warnings if limits exceeded"""
        while self.monitoring:
            try:
                # Check memory usage
                memory_info = self._process.memory_info()
                memory_percent = (memory_info.rss / self.max_memory_bytes) * 100
                
                if memory_percent > self.target_memory_percent:
                    logger.warning(f"Memory usage at {memory_percent:.1f}% of limit")
                    
                # Check CPU usage
                cpu_percent = self._process.cpu_percent(interval=1)
                if cpu_percent > self.target_cpu_percent:
                    logger.warning(f"CPU usage at {cpu_percent:.1f}%")
                    
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                
            time.sleep(5)  # Check every 5 seconds
            
    def set_memory_limit(self):
        """Set hard memory limit for the process"""
        try:
            # Set soft and hard limits
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.max_memory_bytes, self.max_memory_bytes)
            )
            logger.info(f"Memory limit set to {MAX_MEMORY_GB}GB")
        except Exception as e:
            logger.warning(f"Could not set memory limit: {e}")
            
    def get_current_usage(self) -> dict:
        """Get current resource usage"""
        memory_info = self._process.memory_info()
        return {
            'memory_mb': memory_info.rss / (1024 * 1024),
            'memory_percent': (memory_info.rss / self.max_memory_bytes) * 100,
            'cpu_percent': self._process.cpu_percent(interval=0.1),
            'num_threads': self._process.num_threads()
        }
        
    def check_resources_available(self, required_memory_mb: int = 100) -> bool:
        """Check if enough resources are available for a task"""
        current = self.get_current_usage()
        
        # Check memory
        available_memory_mb = (self.max_memory_bytes / (1024 * 1024)) - current['memory_mb']
        if available_memory_mb < required_memory_mb:
            logger.warning(f"Insufficient memory: {available_memory_mb:.1f}MB available, {required_memory_mb}MB required")
            return False
            
        # Check CPU (simple threshold)
        if current['cpu_percent'] > 90:
            logger.warning(f"CPU usage too high: {current['cpu_percent']:.1f}%")
            return False
            
        return True
        
    def wait_for_resources(self, required_memory_mb: int = 100, timeout: int = 60) -> bool:
        """Wait for resources to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.check_resources_available(required_memory_mb):
                return True
            time.sleep(1)
            
        return False


class ResourceContext:
    """Context manager for resource-limited operations"""
    
    def __init__(self, resource_manager: ResourceManager, memory_mb: int = 100):
        self.resource_manager = resource_manager
        self.memory_mb = memory_mb
        
    def __enter__(self):
        if not self.resource_manager.wait_for_resources(self.memory_mb):
            raise RuntimeError("Insufficient resources available")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log final resource usage
        usage = self.resource_manager.get_current_usage()
        logger.debug(f"Task completed. Memory: {usage['memory_mb']:.1f}MB, CPU: {usage['cpu_percent']:.1f}%")