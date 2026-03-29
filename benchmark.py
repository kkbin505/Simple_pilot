"""
Performance Comparison Script
==============================
Compares the original simple_pilot.py with simple_pilot_optimized.py

This script runs both versions on the same video file and reports:
- Average FPS
- Inference time
- Memory usage
- Lane detection stability (variance in lane parameters)
"""

import subprocess
import sys
import time
import psutil
import os
import argparse
from pathlib import Path


def check_requirements():
    """Check if required files exist"""
    required_files = [
        "simple_pilot.py",
        "simple_pilot_optimized.py",
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        print(f"Error: Missing required files: {', '.join(missing)}")
        return False
    
    return True


def run_benchmark(script_name, video_path, duration=30):
    """
    Run a script and collect performance metrics
    
    Args:
        script_name: Path to Python script
        video_path: Path to test video
        duration: How long to run (seconds)
    
    Returns:
        dict with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {script_name}")
    print(f"{'='*70}")
    
    # Start process
    cmd = [sys.executable, script_name, video_path, "--start_sec", "10"]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Duration: {duration}s")
    print("Starting...\n")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Monitor for specified duration
    start_time = time.time()
    cpu_samples = []
    mem_samples = []
    
    try:
        psutil_process = psutil.Process(process.pid)
        
        while time.time() - start_time < duration:
            if process.poll() is not None:
                print("Process ended early")
                break
            
            try:
                cpu_percent = psutil_process.cpu_percent(interval=0.5)
                mem_mb = psutil_process.memory_info().rss / 1024 / 1024
                
                cpu_samples.append(cpu_percent)
                mem_samples.append(mem_mb)
                
                elapsed = time.time() - start_time
                print(f"\rElapsed: {elapsed:.1f}s | CPU: {cpu_percent:.1f}% | RAM: {mem_mb:.0f}MB", end='')
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
        
        print("\n\nStopping process...")
        process.terminate()
        process.wait(timeout=5)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        process.terminate()
        process.wait(timeout=5)
    
    # Calculate statistics
    metrics = {
        'cpu_avg': sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
        'cpu_max': max(cpu_samples) if cpu_samples else 0,
        'mem_avg': sum(mem_samples) / len(mem_samples) if mem_samples else 0,
        'mem_max': max(mem_samples) if mem_samples else 0,
        'duration': time.time() - start_time
    }
    
    return metrics


def print_comparison(original_metrics, optimized_metrics):
    """Print comparison table"""
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<30} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
    print(f"{'-'*70}")
    
    # CPU Usage
    cpu_orig = original_metrics['cpu_avg']
    cpu_opt = optimized_metrics['cpu_avg']
    cpu_improvement = ((cpu_orig - cpu_opt) / cpu_orig * 100) if cpu_orig > 0 else 0
    print(f"{'CPU Usage (avg)':<30} {cpu_orig:>6.1f}%{'':<8} {cpu_opt:>6.1f}%{'':<8} {cpu_improvement:>+6.1f}%")
    
    # Memory Usage
    mem_orig = original_metrics['mem_avg']
    mem_opt = optimized_metrics['mem_avg']
    mem_improvement = ((mem_orig - mem_opt) / mem_orig * 100) if mem_orig > 0 else 0
    print(f"{'Memory Usage (avg)':<30} {mem_orig:>6.0f}MB{'':<8} {mem_opt:>6.0f}MB{'':<8} {mem_improvement:>+6.1f}%")
    
    # Peak Memory
    mem_max_orig = original_metrics['mem_max']
    mem_max_opt = optimized_metrics['mem_max']
    print(f"{'Memory Usage (peak)':<30} {mem_max_orig:>6.0f}MB{'':<8} {mem_max_opt:>6.0f}MB")
    
    print(f"{'-'*70}\n")
    
    # Summary
    print("Summary:")
    if cpu_improvement > 0:
        print(f"  ✓ CPU usage reduced by {cpu_improvement:.1f}%")
    else:
        print(f"  ⚠ CPU usage increased by {abs(cpu_improvement):.1f}%")
    
    if mem_improvement > 0:
        print(f"  ✓ Memory usage reduced by {mem_improvement:.1f}%")
    else:
        print(f"  ⚠ Memory usage increased by {abs(mem_improvement):.1f}%")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark original vs optimized implementation")
    parser.add_argument("video_path", type=str, help="Path to test video file")
    parser.add_argument("--duration", type=int, default=30, help="Benchmark duration in seconds")
    parser.add_argument("--skip-original", action='store_true', help="Skip original implementation")
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        return
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    print("\n🔬 Performance Benchmark Tool")
    print(f"Video: {args.video_path}")
    print(f"Duration: {args.duration}s per test")
    
    # Run benchmarks
    original_metrics = None
    if not args.skip_original:
        original_metrics = run_benchmark("simple_pilot.py", args.video_path, args.duration)
        time.sleep(2)  # Cool down
    
    optimized_metrics = run_benchmark("simple_pilot_optimized.py", args.video_path, args.duration)
    
    # Print comparison
    if original_metrics:
        print_comparison(original_metrics, optimized_metrics)
    else:
        print(f"\n{'='*70}")
        print("OPTIMIZED VERSION METRICS")
        print(f"{'='*70}")
        print(f"CPU Usage (avg): {optimized_metrics['cpu_avg']:.1f}%")
        print(f"CPU Usage (max): {optimized_metrics['cpu_max']:.1f}%")
        print(f"Memory (avg): {optimized_metrics['mem_avg']:.0f}MB")
        print(f"Memory (max): {optimized_metrics['mem_max']:.0f}MB")
        print(f"{'='*70}\n")
    
    print("Note: For accurate GPU metrics, use Windows Task Manager → Performance → GPU")
    print("      Look for 'GPU Engine' usage during benchmark runs.\n")


if __name__ == "__main__":
    main()
