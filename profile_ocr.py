#!/usr/bin/env python3
"""
Profile the manga-ocr model and transformers_ocr.py script to identify performance bottlenecks.
"""

import os
import sys
import time
import argparse
import cProfile
import pstats
import io
from pathlib import Path
import importlib.util
import traceback

def print_color(text, color):
    """Print colored text to console."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")

def check_imports():
    """Check if required packages are installed."""
    required_packages = ["manga_ocr", "torch", "transformers", "psutil"]
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_color(f"✓ {package} is available", "green")
        except ImportError:
            print_color(f"✗ {package} is not installed", "red")
            missing_packages.append(package)
    
    if missing_packages:
        print_color(f"\nMissing packages: {', '.join(missing_packages)}", "red")
        print_color("Install them with: pip install " + " ".join(missing_packages), "yellow")
        return False
    return True

def profile_ocr_model(test_image=None, iterations=1):
    """Profile the manga-ocr model directly."""
    try:
        from manga_ocr import MangaOcr
        import torch
        import psutil
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print_color(f"Initial memory usage: {initial_memory:.2f} MB", "blue")
        
        # Profile model initialization
        print_color("\nProfiling model initialization...", "cyan")
        start_time = time.time()
        
        # Use cProfile for detailed profiling
        pr = cProfile.Profile()
        pr.enable()
        
        mocr = MangaOcr()
        
        pr.disable()
        
        # Print profiling results for initialization
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Show top 20 functions
        print(s.getvalue())
        
        init_time = time.time() - start_time
        print_color(f"Model initialization time: {init_time:.2f} seconds", "green")
        
        # Get memory usage after initialization
        post_init_memory = process.memory_info().rss / 1024 / 1024  # MB
        print_color(f"Memory usage after init: {post_init_memory:.2f} MB (increased by {post_init_memory - initial_memory:.2f} MB)", "blue")
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            print_color(f"GPU memory allocated: {gpu_memory:.2f} MB", "magenta")
        
        # Profile inference if a test image is provided
        if test_image:
            if not os.path.exists(test_image):
                print_color(f"Test image not found: {test_image}", "red")
                return
            
            print_color(f"\nProfiling OCR inference on {test_image} ({iterations} iterations)...", "cyan")
            
            # Warm-up run (not profiled)
            mocr(test_image)
            
            # Start profiling inference
            inference_times = []
            pr = cProfile.Profile()
            pr.enable()
            
            for i in range(iterations):
                start_inference = time.time()
                result = mocr(test_image)
                inference_time = time.time() - start_inference
                inference_times.append(inference_time)
                print_color(f"Iteration {i+1}: {inference_time:.4f} seconds - Result: {result}", "green")
            
            pr.disable()
            
            # Print profiling results for inference
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Show top 20 functions
            print(s.getvalue())
            
            avg_time = sum(inference_times) / len(inference_times)
            print_color(f"Average inference time: {avg_time:.4f} seconds", "yellow")
            
            # Get memory after inference
            post_inference_memory = process.memory_info().rss / 1024 / 1024  # MB
            print_color(f"Memory usage after inference: {post_inference_memory:.2f} MB", "blue")
            
    except Exception as e:
        print_color(f"Error profiling OCR model: {e}", "red")
        traceback.print_exc()

def profile_transformers_ocr(args=None):
    """Profile the transformers_ocr.py script."""
    # Find transformers_ocr.py
    script_path = Path(__file__).parent / "src" / "transformers_ocr.py"
    if not script_path.exists():
        print_color(f"Script not found: {script_path}", "red")
        return
    
    print_color(f"Profiling script: {script_path}", "cyan")
    
    # Set up the arguments
    if args is None:
        args = ["status"]  # Default to status check
    
    # Import the module
    try:
        spec = importlib.util.spec_from_file_location("transformers_ocr", script_path)
        module = importlib.util.module_from_spec(spec)
        sys.argv = [str(script_path)] + args
        
        # Profile the script execution
        pr = cProfile.Profile()
        pr.enable()
        
        start_time = time.time()
        spec.loader.exec_module(module)
        execution_time = time.time() - start_time
        
        pr.disable()
        
        # Print profiling results
        print_color(f"\nExecution time: {execution_time:.2f} seconds", "green")
        
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Show top 30 functions
        print(s.getvalue())
        
    except Exception as e:
        print_color(f"Error profiling script: {e}", "red")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Profile manga-ocr and transformers_ocr performance")
    parser.add_argument("--model-only", action="store_true", help="Profile only the manga-ocr model")
    parser.add_argument("--script-only", action="store_true", help="Profile only the transformers_ocr.py script")
    parser.add_argument("--test-image", type=str, help="Test image to use for OCR profiling")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for inference testing")
    parser.add_argument("--script-args", nargs="+", help="Arguments to pass to transformers_ocr.py")
    
    args = parser.parse_args()
    
    print_color("=== OCR Profiling Tool ===", "cyan")
    
    # Check imports
    if not check_imports():
        return
    
    # Run appropriate profiling based on arguments
    if not args.script_only:
        print_color("\n=== Profiling manga-ocr model ===", "cyan")
        profile_ocr_model(args.test_image, args.iterations)
    
    if not args.model_only:
        print_color("\n=== Profiling transformers_ocr.py script ===", "cyan")
        profile_transformers_ocr(args.script_args)

if __name__ == "__main__":
    main() 