import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set the log level
logger.setLevel(logging.INFO)

# Create a file handler
handler = logging.FileHandler('logfile.log')

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the formatter for the handler
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

