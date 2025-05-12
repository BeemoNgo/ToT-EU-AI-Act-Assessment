import os

# App settings
APP_TITLE = "AI App Review Analyzer"
APP_DESCRIPTION = "Analyze Google Play Store reviews for AI apps and compare user feedback with developer descriptions."
LAYOUT = "wide"  # Streamlit layout

# API settings
OPENAI_MODEL = "gpt-4.1-nano"  # Model to use for analysis

# Review settings
MAX_REVIEWS_TO_FETCH = 1000  # Limit reviews for performance
MIN_REVIEW_LENGTH = 200  # Filter reviews shorter than this
MAX_SEARCH_RESULTS = 20  # Number of search results to return

# Logging format
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s' 