import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai

load_dotenv()
st.set_page_config(page_title="Chat with Gemini!", page_icon=":brain:", layout="centered")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("API key not found. Please set GOOGLE_API_KEY in your environment.")
    st.stop()

try:
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel('gemini-1.5-flash')  # Updated model
except Exception as e:
    st.error(f"Failed to configure Gemini: {e}")
    st.stop()

def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

st.title("🤖 Gemini - ChatBot")

for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

user_prompt = st.chat_input("Ask Gemini...")
if user_prompt:
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        st.error("Please enter a valid, non-empty text prompt.")
    else:
        st.chat_message("user").markdown(user_prompt)
        try:
            gemini_response = st.session_state.chat_session.send_message(user_prompt)
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)
        except gen_ai.types.BlockedPromptException:
            st.error("Prompt blocked by Gemini safety filters.")
        except google.api_core.exceptions.NotFound as e:
            st.error(f"Resource not found (model or API issue): {e}")
        except gen_ai.types.InvalidArgument as e:
            st.error(f"Invalid argument error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
