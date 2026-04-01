#!/usr/bin/env python3
"""
GitHub push helper script - Commits and pushes fine-tuning results.
"""

import os
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, description=""):
    """Run shell command and handle output."""
    if description:
        logger.info(description)
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        logger.error(f"Command failed: {cmd}")
        logger.error(result.stderr)
        return False
    
    if result.stdout:
        logger.info(result.stdout.strip())
    
    return True

def check_git_status():
    """Check if there are changes to commit."""
    logger.info("\n" + "="*70)
    logger.info("Checking git status...")
    logger.info("="*70)
    
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    
    changes = result.stdout.strip()
    if not changes:
        logger.info("✓ No changes to commit")
        return False
    
    logger.info("Changes detected:")
    for line in changes.split('\n'):
        logger.info(f"  {line}")
    
    return True

def commit_results():
    """Commit fine-tuning results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info("\nAdding results to git...")
    if not run_command("git add results/", "Adding results directory..."):
        return False
    
    commit_msg = f"Add fine-tuning results - {timestamp}"
    logger.info(f"\nCommitting with message: '{commit_msg}'")
    
    if not run_command(
        f'git commit -m "{commit_msg}"',
        description="Committing changes..."
    ):
        return False
    
    return True

def push_to_github():
    """Push commits to GitHub."""
    logger.info("\nPushing to GitHub...")
    
    # Get current branch
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip()
    
    logger.info(f"Branch: {branch}")
    
    if not run_command(
        "git push origin main",
        description="Pushing to remote..."
    ):
        return False
    
    return True

def show_summary():
    """Show push summary."""
    result = subprocess.run(
        ["git", "log", "--oneline", "-5"],
        capture_output=True,
        text=True,
    )
    
    logger.info("\n" + "="*70)
    logger.info("Recent commits (showing 5 most recent):")
    logger.info("="*70)
    logger.info(result.stdout)

def main():
    """Main push workflow."""
    logger.info("GitHub Push Helper")
    logger.info("="*70)
    
    # Check if results exist
    if not Path("results").exists():
        logger.warning("⚠ No results directory found")
        logger.warning("Run finetune.py first to generate results")
        sys.exit(1)
    
    # Check git status
    if not check_git_status():
        logger.info("✓ Everything is up to date with remote")
        show_summary()
        return
    
    # Confirm before push
    logger.info("\n" + "="*70)
    response = input("Do you want to commit and push? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        logger.info("Push cancelled")
        return
    
    # Commit
    if not commit_results():
        logger.error("Failed to commit")
        sys.exit(1)
    
    # Push
    if not push_to_github():
        logger.error("Failed to push to GitHub")
        sys.exit(1)
    
    logger.info("\n" + "="*70)
    logger.info("✓ Successfully pushed to GitHub!")
    logger.info("="*70)
    
    show_summary()
    
    logger.info("\n✅ Done!")

if __name__ == "__main__":
    main()
