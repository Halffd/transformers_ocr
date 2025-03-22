#!/usr/bin/env python3
"""
Optimize memory usage for manga-ocr by applying model quantization 
and other memory optimization techniques.
"""

import os
import sys
import gc
import time
import argparse
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

def measure_memory(func, *args, **kwargs):
    """Measure memory usage before and after function execution."""
    try:
        import psutil
        import torch
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Get initial memory stats
        process = psutil.Process(os.getpid())
        initial_ram = process.memory_info().rss / 1024 / 1024  # MB
        initial_gpu = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0  # MB
        
        # Execute function and measure time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Get final memory stats
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        final_ram = process.memory_info().rss / 1024 / 1024  # MB
        final_gpu = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0  # MB
        
        # Print results
        print_color(f"RAM: {initial_ram:.2f} MB → {final_ram:.2f} MB (Δ {final_ram - initial_ram:.2f} MB)", "blue")
        
        if torch.cuda.is_available():
            print_color(f"GPU: {initial_gpu:.2f} MB → {final_gpu:.2f} MB (Δ {final_gpu - initial_gpu:.2f} MB)", "magenta")
            
        print_color(f"Execution time: {execution_time:.4f} seconds", "yellow")
        return result
        
    except Exception as e:
        print_color(f"Error measuring memory: {e}", "red")
        traceback.print_exc()
        return None

def optimize_manga_ocr(test_image=None, quantize=False, half_precision=False):
    """Apply memory optimization techniques to manga-ocr."""
    try:
        from manga_ocr import MangaOcr
        import torch
        
        # Display optimization settings
        print_color("\n=== OCR Memory Optimization ===", "cyan")
        print_color(f"Running with:", "cyan")
        print_color(f"  - Quantization: {'enabled' if quantize else 'disabled'}", "cyan")
        print_color(f"  - Half precision (FP16): {'enabled' if half_precision else 'disabled'}", "cyan")
        
        # Apply global settings
        # Set environment variables to limit CPU usage
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        os.environ["OPENBLAS_NUM_THREADS"] = "2"
        os.environ["NUMEXPR_NUM_THREADS"] = "2"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
        
        print_color("\n=== Testing Baseline Model ===", "cyan")
        
        # Baseline - standard model
        def init_baseline():
            mocr = MangaOcr()
            return mocr
            
        baseline_model = measure_memory(init_baseline)
        
        # Test inference if a test image is provided
        if test_image and baseline_model:
            if not os.path.exists(test_image):
                print_color(f"Test image not found: {test_image}", "red")
            else:
                print_color(f"\nTesting baseline inference with {test_image}", "cyan")
                results = measure_memory(baseline_model, test_image)
                print_color(f"OCR result: {results}", "green")
                
        # Clear memory
        del baseline_model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        # Apply optimizations
        if half_precision:
            print_color("\n=== Testing Half Precision (FP16) Model ===", "cyan")
            
            # Create optimized version - use FP16 precision if enabled
            def init_half_precision():
                # Need to modify MangaOcr to use half precision 
                # This requires accessing the underlying model
                mocr = MangaOcr()
                
                # Convert to half precision (FP16)
                if hasattr(mocr, 'model'):
                    mocr.model = mocr.model.half()
                    print_color("Successfully converted model to half precision", "green")
                else:
                    print_color("Couldn't access model attribute directly", "yellow")
                    
                return mocr
                
            optimized_model = measure_memory(init_half_precision)
            
            # Test inference if a test image is provided
            if test_image and optimized_model:
                if not os.path.exists(test_image):
                    print_color(f"Test image not found: {test_image}", "red")
                else:
                    print_color(f"\nTesting half precision inference with {test_image}", "cyan")
                    try:
                        results = measure_memory(optimized_model, test_image)
                        print_color(f"OCR result: {results}", "green")
                    except Exception as e:
                        print_color(f"Error during half precision inference: {e}", "red")
                        traceback.print_exc()
                        
            # Clear memory
            del optimized_model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        if quantize:
            print_color("\n=== Testing Quantized Model ===", "cyan")
            
            def init_quantized():
                # Create MangaOcr instance first to access the model
                mocr = MangaOcr()
                
                # Apply quantization to the model if possible
                if hasattr(mocr, 'model'):
                    try:
                        # Quantize the model - dynamic quantization (supported for CPU)
                        quantized_model = torch.quantization.quantize_dynamic(
                            mocr.model, 
                            {torch.nn.Linear, torch.nn.Conv2d}, 
                            dtype=torch.qint8
                        )
                        
                        # Replace the model with the quantized version
                        mocr.model = quantized_model
                        print_color("Successfully quantized model", "green")
                    except Exception as e:
                        print_color(f"Quantization error: {e}", "red")
                        traceback.print_exc()
                else:
                    print_color("Couldn't access model attribute directly", "yellow")
                
                return mocr
                
            quantized_model = measure_memory(init_quantized)
            
            # Test inference if a test image is provided
            if test_image and quantized_model:
                if not os.path.exists(test_image):
                    print_color(f"Test image not found: {test_image}", "red")
                else:
                    print_color(f"\nTesting quantized model inference with {test_image}", "cyan")
                    try:
                        results = measure_memory(quantized_model, test_image)
                        print_color(f"OCR result: {results}", "green")
                    except Exception as e:
                        print_color(f"Error during quantized inference: {e}", "red")
                        traceback.print_exc()
                        
        print_color("\n=== Optimization Testing Complete ===", "cyan")
        print_color("Best practice recommendations:", "green")
        print_color("1. Set OMP_NUM_THREADS=2 and other similar env vars to limit CPU threads", "green")
        print_color("2. Call torch.cuda.empty_cache() after OCR operations", "green")
        print_color("3. Force garbage collection with gc.collect() periodically", "green")
        print_color("4. Consider moving to model quantization for production", "green")
        print_color("5. For GPU use, consider half precision (FP16) to reduce memory usage", "green")
            
    except Exception as e:
        print_color(f"Error optimizing OCR model: {e}", "red")
        traceback.print_exc()

def update_transformers_ocr():
    """Update transformers_ocr.py with optimizations."""
    script_path = Path(__file__).parent / "src" / "transformers_ocr.py"
    if not script_path.exists():
        print_color(f"Script not found: {script_path}", "red")
        return
    
    print_color(f"Analyzing {script_path} for optimization opportunities...", "cyan")
    
    # Read the file
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check if optimizations are already applied
    already_optimized = "OMP_NUM_THREADS" in content and "gc.collect()" in content
    
    if already_optimized:
        print_color("The script already contains memory optimizations.", "yellow")
        return
    
    print_color("Found optimization opportunities:", "green")
    print_color("1. Add environment variable settings to limit CPU threads", "green")
    print_color("2. Add garbage collection calls", "green")
    print_color("3. Add torch.cuda.empty_cache() calls", "green")
    
    print_color("\nTo apply these optimizations, run the provided dev.sh script:", "yellow")
    print_color("  ./dev.sh optimize", "yellow")

def add_optimizations():
    """Add optimization functions to the MangaOcrWrapper class."""
    # This is just a blueprint for what could be added
    optimization_code = """
def optimize_memory():
    # Set environment variables to limit CPU usage
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
def quantize_model(model):
    # Apply quantization to reduce memory usage
    try:
        import torch
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.Conv2d}, 
            dtype=torch.qint8
        )
        return quantized_model
    except Exception as e:
        print(f"Error quantizing model: {e}")
        return model
"""
    print_color("Example optimization functions:", "cyan")
    print(optimization_code)

def main():
    parser = argparse.ArgumentParser(description="Optimize manga-ocr memory usage")
    parser.add_argument("--test-image", type=str, help="Test image to use for OCR testing")
    parser.add_argument("--quantize", action="store_true", help="Enable model quantization (experimental)")
    parser.add_argument("--half-precision", action="store_true", help="Use half precision (FP16)")
    parser.add_argument("--update-script", action="store_true", help="Analyze transformers_ocr.py for optimization")
    parser.add_argument("--show-examples", action="store_true", help="Show example optimization functions")
    
    args = parser.parse_args()
    
    print_color("=== Manga-OCR Memory Optimizer ===", "cyan")
    
    # Check imports
    if not check_imports():
        return
    
    # Run appropriate optimization based on arguments
    if args.update_script:
        update_transformers_ocr()
    elif args.show_examples:
        add_optimizations()
    else:
        optimize_manga_ocr(args.test_image, args.quantize, args.half_precision)

if __name__ == "__main__":
    main() 