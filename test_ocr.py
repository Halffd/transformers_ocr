#!/usr/bin/env python3
# Test script for transformers_ocr.py to help diagnose OCR issues

import os
import sys
import time
import traceback
import subprocess

def main():
    print("=== Transformers OCR Test Script ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Test step 1: Check environment
    print("\n1. Checking environment and dependencies...")
    try:
        # Check if manga-ocr is installed
        try:
            import manga_ocr
            print(f"Found manga_ocr: {manga_ocr.__file__}")
        except ImportError as e:
            print(f"manga_ocr not available: {e}")
            
        # Check PyTorch
        try:
            import torch
            print(f"Found PyTorch: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available() if hasattr(torch, 'cuda') else False}")
        except ImportError as e:
            print(f"PyTorch not available: {e}")
            
        # Check transformers
        try:
            import transformers
            print(f"Found transformers: {transformers.__version__}")
        except ImportError as e:
            print(f"transformers not available: {e}")
    except Exception as e:
        print(f"Error checking environment: {e}")
        traceback.print_exc()
    
    # Test step 2: Check script files
    print("\n2. Checking script files...")
    try:
        script_path = os.path.join("src", "transformers_ocr.py")
        if os.path.exists(script_path):
            print(f"Found transformers_ocr.py: {os.path.abspath(script_path)}")
        else:
            print(f"transformers_ocr.py not found at {script_path}")
    except Exception as e:
        print(f"Error checking script files: {e}")
        traceback.print_exc()

    # Test step 3: Check OCR setup
    print("\n3. Checking OCR service status...")
    try:
        result = subprocess.run([sys.executable, os.path.join("src", "transformers_ocr.py"), "status"], 
                                capture_output=True, text=True)
        print(f"OCR service status: {result.stdout.strip()}")
    except Exception as e:
        print(f"Error checking OCR service status: {e}")
        traceback.print_exc()
    
    # Test step 4: Direct test of OCR
    print("\n4. Testing OCR directly...")
    try:
        # Check if we can initialize the OCR model
        try:
            from manga_ocr import MangaOcr
            print("Initializing MangaOcr...")
            mocr = MangaOcr()
            print("MangaOcr initialized successfully!")
            
            # Try to run OCR on a test image if one is available
            test_images = [
                "test_image.png",
                "test.png",
                os.path.join("src", "test_image.png"),
                os.path.join("src", "test.png"),
            ]
            
            for img_path in test_images:
                if os.path.exists(img_path):
                    print(f"Found test image: {img_path}")
                    print("Running OCR on test image...")
                    start_time = time.time()
                    text = mocr(img_path)
                    end_time = time.time()
                    print(f"OCR completed in {end_time - start_time:.2f}s")
                    print(f"OCR result: {text}")
                    break
            else:
                print("No test images found. Create a test image to test OCR directly.")
        except Exception as e:
            print(f"Error testing OCR directly: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error in OCR test section: {e}")
        traceback.print_exc()
    
    # Test step 5: Full script test (optional)
    print("\n5. Testing full script functionality (if test image available)...")
    try:
        # Try to use the script with a test image if available
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"Testing full script with image: {img_path}")
                result = subprocess.run(
                    [sys.executable, os.path.join("src", "transformers_ocr.py"), 
                     "recognize", "--image-path", img_path, "-d"],
                    capture_output=True, text=True, timeout=60
                )
                print(f"Script output:\n{result.stdout}")
                if result.stderr:
                    print(f"Script errors:\n{result.stderr}")
                break
        else:
            print("Skipping full script test - no test image available")
    except Exception as e:
        print(f"Error in full script test: {e}")
        traceback.print_exc()
    
    print("\n=== Test Complete ===")
    

if __name__ == "__main__":
    main() 