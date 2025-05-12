"""
Logging service for the application.
"""
import logging
import streamlit as st
from config.settings import LOG_FORMAT

def setup_logger(name="app_logger"):
    """Setup and return a logger instance."""
    # Configure the root logger
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    
    # Get logger instance
    logger = logging.getLogger(name)
    return logger

# Create a singleton logger instance
logger = setup_logger()

class StatusLogger:
    """
    A wrapper class for logging to both console and Streamlit status.
    Works with st.status context manager.
    """
    def __init__(self, status_obj=None):
        self.status = status_obj
        self.logger = logger
    
    def set_status(self, status_obj):
        """Set or update the status object."""
        self.status = status_obj
        return self
    
    def info(self, message):
        """Log info message to both logger and status if available."""
        self.logger.info(message)
        if self.status:
            self.status.write(message)
    
    def warning(self, message):
        """Log warning message to both logger and status if available."""
        self.logger.warning(message)
        if self.status:
            self.status.warning(message)
    
    def error(self, message, exc_info=False):
        """Log error message to both logger and status if available."""
        self.logger.error(message, exc_info=exc_info)
        if self.status:
            self.status.error(message)
    
    def update(self, label=None, state=None, expanded=None):
        """Update status object if available."""
        if self.status and label:
            self.logger.info(label)
            kwargs = {}
            if label:
                kwargs["label"] = label
            if state:
                kwargs["state"] = state  
            if expanded is not None:
                kwargs["expanded"] = expanded
            self.status.update(**kwargs)
    
    def write(self, message):
        """Write message to status if available."""
        self.logger.info(message)
        if self.status:
            self.status.write(message) 