import google.generativeai as genai
import os
from dotenv import load_dotenv 
import logging
from datetime import datetime

# --- Logging Setup ---
def setup_logging(model_name="gemini_test"):
    # Get project root (assuming script is in models/gemini/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Saving logs to {log_file}")
    return log_file

setup_logging()

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# UPDATE THIS LINE TO THE CHOSEN MODEL
model = genai.GenerativeModel('gemini-2.5-flash')

try:
    response = model.generate_content("What are the 3 most common visual artifacts in deepfake videos?")
    logging.info(response.text)
except Exception as e:
    logging.error(f"Error: {e}")