import os
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# Configure google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)

def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("models/gemini-1.5-flash")  # Updated to a likely current model
    return gemini_pro_model

# Get response from Gemini-Pro-Vision model - image/text to text
def gemini_pro_vision_response(prompt, image):
    gemini_pro_vision_model = genai.GenerativeModel("models/gemini-1.5-flash")  # Assuming multimodal support
    response = gemini_pro_vision_model.generate_content([prompt, image])
    result = response.text
    return result

# Get response from embeddings model - text to embeddings
def embeddings_model_response(input_text):
    embedding_model = "models/embedding-001"  # Typically correct
    embedding = genai.embed_content(model=embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list

# Get response from Gemini-Pro model - text to text
def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("models/gemini-1.5-flash")  # Updated to a likely current model
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result
