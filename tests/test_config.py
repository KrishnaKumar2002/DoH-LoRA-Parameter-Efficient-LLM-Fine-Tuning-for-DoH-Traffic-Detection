"""
Unit tests for DoH-LoRA configuration module.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from src.doh_lora.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""

    def test_config_initialization(self):
        """Test that Config initializes with default values."""
        # Test that config has required attributes
        self.assertIsInstance(Config.BASE_MODEL, str)
        self.assertIsInstance(Config.SEED, int)
        self.assertIsInstance(Config.DEVICE, str)
        self.assertGreater(Config.EPOCHS, 0)
        self.assertGreater(Config.BATCH_SIZE, 0)

    def test_ensure_dirs(self):
        """Test that ensure_dirs creates necessary directories."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            Config.ensure_dirs()
            # Should call mkdir for RESULTS_DIR and LOGS_DIR
            self.assertGreaterEqual(mock_mkdir.call_count, 2)

    def test_get_summary(self):
        """Test that get_summary returns a dictionary."""
        summary = Config.get_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('base_model', summary)
        self.assertIn('device', summary)


if __name__ == '__main__':
    unittest.main()