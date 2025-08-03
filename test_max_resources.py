#!/usr/bin/env python3
"""Test maximum resource utilization for OCR system"""

import asyncio
import time
import psutil
import os
import sys
from pathlib import Path
from cpu_only_ocr_system import CPUOnlyOCRSystem, get_resource_info

# Test configuration
TEST_DIR = Path("test_documents")
OUTPUT_DIR = Path("output/max_resource_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ResourceMonitor:
    """Monitor resource usage during tests"""
    
    def __init__(self):
        self.max_cpu = 0
        self.max_ram = 0
        self.samples = []
        self.monitoring = True
        
    async def monitor(self):
        """Monitor resources every 0.5 seconds"""
        while self.monitoring:
            info = get_resource_info()
            self.max_cpu = max(self.max_cpu, info['cpu_percent'])
            self.max_ram = max(self.max_ram, info['ram_percent'])
            self.samples.append({
                'time': time.time(),
                'cpu': info['cpu_percent'],
                'ram': info['ram_percent'],
                'ram_gb': info['ram_gb']
            })
            await asyncio.sleep(0.5)
            
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        
    def get_report(self):
        """Get resource usage report"""
        return {
            'max_cpu_percent': self.max_cpu,
            'max_ram_percent': self.max_ram,
            'avg_cpu_percent': sum(s['cpu'] for s in self.samples) / len(self.samples) if self.samples else 0,
            'avg_ram_percent': sum(s['ram'] for s in self.samples) / len(self.samples) if self.samples else 0,
            'total_samples': len(self.samples)
        }

async def test_single_large_file():
    """Test processing a large file"""
    print("\n=== Testing Single Large File ===")
    
    ocr = CPUOnlyOCRSystem()
    monitor = ResourceMonitor()
    
    # Start monitoring
    monitor_task = asyncio.create_task(monitor.monitor())
    
    # Find largest PDF in test directory
    pdf_files = list(TEST_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in test directory")
        return
        
    largest_pdf = max(pdf_files, key=lambda p: p.stat().st_size)
    print(f"Processing: {largest_pdf.name} ({largest_pdf.stat().st_size / 1024 / 1024:.2f} MB)")
    
    start = time.time()
    try:
        result = await ocr.process_file(str(largest_pdf))
        print(f"✅ Successfully processed in {time.time() - start:.2f}s")
        print(f"   Text length: {len(result.text)} characters")
        print(f"   Tables found: {len(result.tables)}")
        print(f"   CPU usage during processing: {result.cpu_usage:.1f}%")
        print(f"   RAM usage during processing: {result.ram_usage_mb:.1f} MB")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        monitor.stop()
        await monitor_task
        
    report = monitor.get_report()
    print(f"\n📊 Resource Usage Report:")
    print(f"   Max CPU: {report['max_cpu_percent']:.1f}%")
    print(f"   Max RAM: {report['max_ram_percent']:.1f}%")
    print(f"   Avg CPU: {report['avg_cpu_percent']:.1f}%")
    print(f"   Avg RAM: {report['avg_ram_percent']:.1f}%")

async def test_parallel_processing():
    """Test processing multiple files in parallel"""
    print("\n=== Testing Parallel Processing (Maximum Load) ===")
    
    ocr = CPUOnlyOCRSystem()
    monitor = ResourceMonitor()
    
    # Get all test files
    test_files = []
    for ext in ['*.pdf', '*.png', '*.jpg', '*.docx']:
        test_files.extend(TEST_DIR.glob(ext))
    
    if not test_files:
        print("No test files found")
        return
        
    print(f"Processing {len(test_files)} files in parallel...")
    
    # Start monitoring
    monitor_task = asyncio.create_task(monitor.monitor())
    
    start = time.time()
    try:
        results = await ocr.process_batch([str(f) for f in test_files[:10]])  # Process up to 10 files
        successful = sum(1 for r in results if r.text)
        print(f"✅ Processed {successful}/{len(results)} files in {time.time() - start:.2f}s")
        
        # Print individual results
        for i, (f, r) in enumerate(zip(test_files[:10], results)):
            if r.text:
                print(f"   {f.name}: {len(r.text)} chars")
            else:
                print(f"   {f.name}: Failed")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        monitor.stop()
        await monitor_task
        
    report = monitor.get_report()
    print(f"\n📊 Resource Usage Report:")
    print(f"   Max CPU: {report['max_cpu_percent']:.1f}%")
    print(f"   Max RAM: {report['max_ram_percent']:.1f}%")
    print(f"   Avg CPU: {report['avg_cpu_percent']:.1f}%")
    print(f"   Avg RAM: {report['avg_ram_percent']:.1f}%")

async def test_memory_stress():
    """Test memory handling with large images"""
    print("\n=== Testing Memory Stress (Large Images) ===")
    
    ocr = CPUOnlyOCRSystem()
    monitor = ResourceMonitor()
    
    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(TEST_DIR.glob(ext))
    
    if not image_files:
        print("No image files found")
        return
        
    print(f"Processing {len(image_files)} images sequentially...")
    
    # Start monitoring
    monitor_task = asyncio.create_task(monitor.monitor())
    
    start = time.time()
    errors = 0
    
    for img_file in image_files:
        try:
            print(f"  Processing {img_file.name}...", end='', flush=True)
            result = await ocr.process_file(str(img_file))
            print(f" ✅ ({result.processing_time:.2f}s)")
        except Exception as e:
            print(f" ❌ Error: {e}")
            errors += 1
            
    monitor.stop()
    await monitor_task
    
    print(f"\nCompleted in {time.time() - start:.2f}s with {errors} errors")
    
    report = monitor.get_report()
    print(f"\n📊 Resource Usage Report:")
    print(f"   Max CPU: {report['max_cpu_percent']:.1f}%")
    print(f"   Max RAM: {report['max_ram_percent']:.1f}%")
    print(f"   Avg CPU: {report['avg_cpu_percent']:.1f}%")
    print(f"   Avg RAM: {report['avg_ram_percent']:.1f}%")

async def main():
    """Run all resource tests"""
    print("🚀 OCR Maximum Resource Utilization Test")
    print("=" * 50)
    
    # Show current configuration
    print(f"\nConfiguration:")
    print(f"  MAX_CPU_PERCENT: 95%")
    print(f"  MAX_RAM_PERCENT: 90%")
    print(f"  MAX_WORKERS: {os.cpu_count() if os.cpu_count() <= 8 else 8}")
    
    # Show system info
    info = get_resource_info()
    print(f"\nSystem Info:")
    print(f"  CPU Cores: {info['cpu_count']}")
    print(f"  RAM: {info['ram_gb']:.2f} GB")
    print(f"  Current CPU: {info['cpu_percent']:.1f}%")
    print(f"  Current RAM: {info['ram_percent']:.1f}%")
    
    # Run tests
    await test_single_large_file()
    await test_parallel_processing()
    await test_memory_stress()
    
    print("\n✅ All tests completed!")
    print("\nNote: The system is configured to use up to 95% CPU and 90% RAM")
    print("for maximum performance while maintaining stability.")

if __name__ == "__main__":
    asyncio.run(main())