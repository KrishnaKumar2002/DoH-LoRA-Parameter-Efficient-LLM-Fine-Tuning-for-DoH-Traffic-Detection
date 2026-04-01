#!/usr/bin/env python3
"""
Fine-tuning execution script with efficient quantization.
Handles setup, execution, and results preparation for GitHub push.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_optional_dependencies():
    """Install optional dependencies for efficient fine-tuning."""
    logger.info("Installing optional dependencies for efficient training...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "bitsandbytes==0.41.3"],
            check=False,
        )
        logger.info("✓ Optional dependencies installed")
    except Exception as e:
        logger.warning(f"Could not install bitsandbytes: {e}")
        logger.warning("  Training will continue without 8-bit quantization")

def check_data_files():
    """Verify required CSV files exist."""
    logger.info("Checking data files...")
    
    required_files = [
        "data/merge_first_layer.csv",
        "data/merge_second_layer.csv",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"✗ Missing file: {file_path}")
            return False
        logger.info(f"✓ Found: {file_path}")
    
    return True

def run_finetuning():
    """Run the fine-tuning pipeline."""
    logger.info("=" * 70)
    logger.info("Starting DoH-LoRA Fine-Tuning Pipeline")
    logger.info("=" * 70)
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from doh_lora.pipeline import run_pipeline
        run_pipeline()
        
        logger.info("=" * 70)
        logger.info("✓ Fine-tuning completed successfully!")
        logger.info("=" * 70)
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all dependencies are installed:")
        logger.error("  pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"Fine-tuning error: {e}")
        import traceback
        traceback.print_exc()
        return False

def prepare_for_github():
    """Prepare results for GitHub push."""
    logger.info("\n" + "=" * 70)
    logger.info("Preparing results for GitHub")
    logger.info("=" * 70)
    
    results_dir = Path("results")
    
    if not results_dir.exists():
        logger.warning("No results directory found")
        return
    
    # List important result files
    logger.info("\n📊 Generated results:")
    for file_path in sorted(results_dir.rglob("*")):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / 1024 / 1024
            logger.info(f"  • {file_path.relative_to(results_dir):<50} ({size_mb:.2f} MB)")
    
    logger.info("\n" + "=" * 70)
    logger.info("Ready for GitHub push!")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("  1. git add results/")
    logger.info("  2. git commit -m 'Add fine-tuning results'")
    logger.info("  3. git push origin main")
    logger.info("\nOR use: python github_push.py")

def main():
    """Main execution flow."""
    logger.info("DoH-LoRA Fine-Tuning System")
    logger.info("Version 1.0.0 - With Efficient Quantization")
    logger.info("")
    
    # Check environment
    if not check_data_files():
        logger.error("\n❌ Data files missing. Cannot proceed.")
        sys.exit(1)
    
    # Install optional dependencies
    install_optional_dependencies()
    
    # Run fine-tuning
    success = run_finetuning()
    
    if not success:
        logger.error("\n❌ Fine-tuning failed")
        sys.exit(1)
    
    # Prepare for GitHub
    prepare_for_github()
    
    logger.info("\n✅ All done!")

if __name__ == "__main__":
    main()
