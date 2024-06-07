import unittest
from src.config import Config

class TestConfig(unittest.TestCase):

    def test_config_attributes(self):
        self.assertTrue(hasattr(Config, 'RAW_DATA_DIR'))
        self.assertTrue(hasattr(Config, 'PROCESSED_DATA_DIR'))
        self.assertTrue(hasattr(Config, 'BASELINE_MODEL_DIR'))
        self.assertTrue(hasattr(Config, 'FINE_TUNED_MODEL_DIR'))
        self.assertTrue(hasattr(Config, 'EPOCHS'))
        self.assertTrue(hasattr(Config, 'LEARNING_RATE'))

if __name__ == '__main__':
    unittest.main()
