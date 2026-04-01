#!/usr/bin/env python3
"""
Main entry point for DoH-LoRA fine-tuning application.

Usage:
    python -m src.doh_lora.main

Environment variables:
    FIRST_LAYER_CSV: Path to first layer CSV (default: data/merge_first_layer.csv)
    SECOND_LAYER_CSV: Path to second layer CSV (default: data/merge_second_layer.csv)
    LOG_LEVEL: Logging level (default: INFO)
"""

import sys

from .pipeline import run_pipeline


def main():
    """Entry point for the application."""
    try:
        run_pipeline()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
