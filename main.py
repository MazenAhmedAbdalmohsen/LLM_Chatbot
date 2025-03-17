import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file or Streamlit Cloud secrets.")
    st.stop()
genai.configure(api_key=api_key)

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Gemini AI",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu('Gemini AI',
                           ['ChatBot', 'Image Captioning', 'Embed text', 'Ask me anything'],
                           menu_icon='robot', icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
                           default_index=0)

def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

if selected == 'ChatBot':
    model = genai.GenerativeModel("gemini-1.0-pro")  # Updated model name
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    st.title("🤖 ChatBot")
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    user_prompt = st.chat_input("Ask Gemini...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        try:
            gemini_response = st.session_state.chat_session.send_message(user_prompt)
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)
        except Exception as e:
            st.error(f"Error getting response from Gemini: {str(e)}")

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
        try:
            model = genai.GenerativeModel("gemini-1.0-pro-vision")  # Updated vision model
            caption = model.generate_content([default_prompt, image]).text
            with col2:
                st.info(caption)
        except Exception as e:
            st.error(f"Error generating caption: {str(e)}")

if selected == "Embed text":
    st.title("🔡 Embed Text")
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")
    if st.button("Get Response") and user_prompt:
        try:
            response = genai.embed_content(model="embedding-001", content=user_prompt)
            st.markdown(str(response))
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")

if selected == "Ask me anything":
    st.title("❓ Ask me a question")
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")
    if st.button("Get Response") and user_prompt:
        try:
            model = genai.GenerativeModel("gemini-1.0-pro")  # Updated model name
            response = model.generate_content(user_prompt).text
            st.markdown(response)
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
