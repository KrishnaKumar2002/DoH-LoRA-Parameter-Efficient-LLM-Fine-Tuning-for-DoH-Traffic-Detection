#!/usr/bin/env python3
"""
Wrapper script to run the DoH-LoRA fine-tuning pipeline with efficient techniques.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Run the pipeline
if __name__ == "__main__":
    try:
        from doh_lora.pipeline import run_pipeline
        print("\n" + "="*70)
        print("DoH-LoRA Fine-Tuning Pipeline - Running")
        print("="*70)
        run_pipeline()
        print("\n" + "="*70)
        print("Pipeline Complete!")
        print("="*70)
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
