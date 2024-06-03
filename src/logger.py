import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger"""
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Example of using the logger
if not os.path.exists('../logs'):
    os.makedirs('../logs')

logger = setup_logger('main_logger', '../logs/main.log')
logger.info('Logger setup complete.')
