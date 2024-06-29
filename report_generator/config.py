import os
from dotenv import load_dotenv
import pybliometrics.scopus
import voyageai

def set_api_keys(pybliometrics_key: str = None, voyage_key: str = None, together_key: str = None):
    """
    Set API keys for the various services used by the ReportGenerator.
    If keys are not provided, it will attempt to load them from environment variables.
    
    Returns:
        bool: True if all keys are set successfully, False otherwise.
    """
    load_dotenv()  # Load environment variables from .env file if it exists

    if pybliometrics_key:
        os.environ['PYBLIOMETRICS_API_KEY'] = pybliometrics_key
    if voyage_key:
        os.environ['VOYAGE_API_KEY'] = voyage_key
    if together_key:
        os.environ['TOGETHER_API_KEY'] = together_key

    try:
        # Initialize Scopus with the API key
        pybliometrics.scopus.init()

        # Set Voyage AI API key
        voyageai.api_key = os.getenv('VOYAGE_API_KEY')

        # Set Together API key (if you decide to handle it here)
        # together.api_key = os.getenv('TOGETHER_API_KEY')

        return True
    except Exception as e:
        print(f"Error setting API keys: {e}")
        return False