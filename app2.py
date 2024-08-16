import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from PIL import Image
import io
import base64

# Set up OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Set up Streamlit app
st.title("Image and Text Analysis with GPT-4-Vision")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type="png")

# Text input for prompt
user_prompt = st.text_input("Enter your prompt:")

if uploaded_file is not None and user_prompt:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Encode the image
    base64_image = encode_image(image)

    # Set up ChatOpenAI model
    chat = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=300)

    # Prepare messages
    messages = [
        SystemMessage(content="You are an AI assistant capable of analyzing images and text."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": f"Analyze the following image and respond to the user's prompt: {user_prompt}"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        ])
    ]

    # Get the response
    response = chat(messages)

    # Display the result
    st.subheader("Analysis Result:")
    st.write(response.content)
else:
    st.write("Please upload an image and enter a prompt to get started.")