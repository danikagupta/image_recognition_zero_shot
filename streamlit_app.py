import streamlit as st
from openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
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

def run_gpt4_vision(prompt, image):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"}
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

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

    # Create prompt template
    template = """
    You are an AI assistant capable of analyzing images and text.
    Analyze the following image and respond to the user's prompt:

    User prompt: {user_prompt}

    Image: {image}

    Please provide a detailed analysis based on the image and the user's prompt.
    """

    prompt = PromptTemplate(
        input_variables=["user_prompt", "image"],
        template=template,
    )

    # Set up LLMChain
    #llm = OpenAI(temperature=0.7, model_name="gpt-4o-mini")
    #chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain
    #response = chain.run(user_prompt=user_prompt, image=base64_image)
    response = run_gpt4_vision(prompt.format(user_prompt=user_prompt, image="[Image]"), base64_image)


    # Display the result
    st.subheader("Analysis Result:")
    st.write(response)
else:
    st.write("Please upload an image and enter a prompt to get started.")
