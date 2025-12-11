import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import logging
from datetime import datetime

# --- Logging Setup ---
def setup_logging(model_name="gemini_detect"):
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

# 1. Setup
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def analyze_image(image_path):
    logging.info(f"🚀 Loading image: {image_path}...")
    
    try:
        # Load image with Pillow
        img = Image.open(image_path)
    except Exception as e:
        logging.error(f"❌ Could not open image: {e}")
        return

    # 2. The "Forensic" Prompt
    # Deepfakes often fail at geometry, symmetry, and text.
    prompt = """
    Act as an expert digital forensics analyst. Examine this image for signs of AI generation or manipulation (Deepfake).
    
    Scrutinize the following specific areas:
    1. **Eyes & Pupils:** Are reflections symmetrical? Are pupils perfectly circular?
    2. **Hands & Fingers:** Are there extra fingers, strange joints, or merging skin?
    3. **Background:** Is there warping, nonsensical text, or impossible architecture?
    4. **Accessories:** Do earrings, glasses, or collars match perfectly on both sides?
    5. **Skin Texture:** Is it overly smooth (plastic-like) or does it have realistic pores?

    Provide a verdict: "LIKELY REAL" or "LIKELY FAKE" with a confidence score and a list of suspicious artifacts found.
    """

    # 3. Send to Gemini 2.5 Flash
    # Note: Flash is excellent at spotting visual inconsistencies rapidly
    logging.info("👀 Analyzing pixel artifacts...")
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    try:
        response = model.generate_content([prompt, img])
        
        logging.info("\n" + "="*40)
        logging.info("🕵️ IMAGE FORENSICS REPORT")
        logging.info("="*40)
        logging.info(response.text)
    except Exception as e:
        logging.error(f"API Error: {e}")

if __name__ == "__main__":
    setup_logging()
    # Example usage - you might want to make this configurable or scan a directory
    # analyze_image("path/to/image.jpg")
    pass
if __name__ == "__main__":
    # Make sure you have an image in your data folder!
    # You can use .jpg, .png, or .webp
    target_image = "data/test_image.jpg" 
    
    if os.path.exists(target_image):
        analyze_image(target_image)
    else:
        print(f"❌ File not found: {target_image}")
        print("Please put a 'test_image.jpg' inside your 'data' folder.")