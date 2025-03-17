import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import google.generativeai as genai  # Import added for explicit configuration

from gemini_utility import (
    load_gemini_pro_model,
    gemini_pro_response,
    gemini_pro_vision_response,
    embeddings_model_response
)

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file or Streamlit Cloud secrets.")
    st.stop()
genai.configure(api_key=api_key)  # Configure the API key for all Gemini interactions

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Gemini AI",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu('Gemini AI',
                           ['ChatBot',
                            'Image Captioning',
                            'Embed text',
                            'Ask me anything'],
                           menu_icon='robot', icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
                           default_index=0
                           )

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Chatbot page
if selected == 'ChatBot':
    try:
        model = load_gemini_pro_model()
    except Exception as e:
        st.error(f"Failed to load Gemini model: {str(e)}")
        st.stop()

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page
    st.title("🤖 ChatBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Gemini-Pro...")
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        try:
            gemini_response = st.session_state.chat_session.send_message(user_prompt)
            # Display Gemini-Pro's response
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)
        except Exception as e:
            st.error(f"Error getting response from Gemini: {str(e)}")

# Image captioning page
if selected == "Image Captioning":
    st.title("📷 Snap Narrate")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption") and uploaded_image:
        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((800, 500))
            st.image(resized_img)

        default_prompt = "Write a short caption for this image"

        # Get the caption of the image from the gemini-pro-vision LLM
        try:
            caption = gemini_pro_vision_response(default_prompt, image)
            with col2:
                st.info(caption)
        except Exception as e:
            st.error(f"Error generating caption: {str(e)}")

# Text embedding model
if selected == "Embed text":
    st.title("🔡 Embed Text")

    # Text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if st.button("Get Response") and user_prompt:
        try:
            response = embeddings_model_response(user_prompt)
            st.markdown(response)
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")

# Ask me anything page
if selected == "Ask me anything":
    st.title("❓ Ask me a question")

    # Text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response") and user_prompt:
        try:
            response = gemini_pro_response(user_prompt)
            st.markdown(response)
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
