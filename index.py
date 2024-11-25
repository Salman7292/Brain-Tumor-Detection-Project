import requests
from io import BytesIO
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.image as mpimg
import os
from tensorflow.keras.models import load_model
import json


import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


st.set_page_config(
    page_icon="Logo3.png",
    page_title="Brain Tumor Detection | app",
    layout="wide"
)


# # Load the saved model
cnn_model = load_model('Brain_Tumor_Detection_Model.h5')



genai.configure(api_key="AIzaSyAkgJ60JabYJzxHTDqA_VwD6M_ptR0s5XU")



def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def to_markdown2(list1):
  list2=list()
  for i in list1:
    if "*" in i:
      newword=i.replace("*"," ")
      list2.append(newword)
    elif "-" in i:
        newword=i.replace("-"," ")
        list2.append(newword)



  return list2


model = genai.GenerativeModel('gemini-pro')

# genmini api



def generate_tumor_info(tumor_label):
    prompt = f"""
    You are given the following brain tumor label:

    {tumor_label}

    Your task is to provide detailed information about the tumor type associated with the label. Specifically, you need to:
    1. Describe the tumor type associated with the label.
    2. List and explain the common symptoms of this tumor.
    3. Provide a summary of the best available treatments for this tumor type.
    4. Suggest references or specialists who are known for treating this specific type of tumor.

    The tumor labels are:
    - 'pituitary_tumor'
    - 'glioma_tumor'
    - 'no_tumor'
    - 'meningioma_tumor'

    Based on the provided label, generate the following information:

    **Tumor Description:**
    [Detailed description of the tumor type]

    **Symptoms:**
    [Detailed list of symptoms associated with the tumor]

    **Treatments:**
    [Summary of best treatments available for this tumor type]

    **Specialist References:**
    [Suggestions for specialists or institutions known for treating this tumor type]

    Ensure that:
    - The information is accurate and up-to-date.
    - Symptoms and treatments are described in detail.
    - Specialist recommendations are relevant and based on current medical practices.

    If the label provided is 'no_tumor', provide a congratulatory message stating: "Congratulations! No tumor detected. Everything appears to be normal."

    """
    response = model.generate_content(prompt)
    return response








# Class labels
labels = ['pituitary_tumor', 'glioma_tumor', 'no_tumor', 'meningioma_tumor']

def image_preprocessing(image,target_size=(150, 150)):
        # Load the image
    img = Image.open(image)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # # # Scale the image values to [0, 1]
    # img_array = img_array.astype('float32') / 255.

    predictions = cnn_model.predict(img_array)

    indices = predictions.argmax()


    class_name=labels[indices]
    


    return class_name








def image_preprocessing_from_url(image_url, target_size=(150, 150)):
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Convert the image to RGB to ensure 3 color channels
        img = img.convert("RGB")
        
        # Resize the image
        img = img.resize(target_size)

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Add batch dimension (1, 150, 150, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # # Scale the image values to [0, 1]
        # img_array = img_array.astype('float32') / 255.

        # Predict using the preloaded CNN model
        predictions = cnn_model.predict(img_array)

        # Get the index of the class with the highest probability
        indices = predictions.argmax()

        # Get the class name from the labels list
        class_name = labels[indices]

        return class_name
    
    except Exception as e:
        return f"An error occurred during image preprocessing: {e}"    
    

# Custom CSS to style the BMI result box
st.markdown("""
    <style>
.bmiresult {
    font-family: Arial, sans-serif;
    background-color: #ffffff;
    border: 2px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    text-align: center;
    width: 465px;
    /* HEIGHT: 328PX; */
    margin: auto;
}

.bmiresult h1 {
    margin: 0 0 10px;
    color: #333;
    font-size: 34px;
}

    .bmiresult h3 {
        margin: 0;
        color: #007BFF;
        font-size: 2em;
    }

    .bmiresult p {
        margin-top: 10px;
        color: #333;
    }
            
      
    </style>
""", unsafe_allow_html=True)

# CSS styling for the Streamlit app
page_bg_img = f"""
<style>
[data-testid="stSidebar"] > div:first-child {{
    background-repeat: no-repeat;
    background-attachment: fixed;
    background: rgb(18 18 18 / 0%);
}}

.st-emotion-cache-1gv3huu {{
    position: relative;
    top: 2px;
    background-color: #000;
    z-index: 999991;
    min-width: 244px;
    max-width: 550px;
    transform: none;
    transition: transform 300ms, min-width 300ms, max-width 300ms;
}}

.st-emotion-cache-1jicfl2 {{
    width: 100%;
    padding: 4rem 1rem 4rem;
    min-width: auto;
    max-width: initial;

}}


.st-emotion-cache-4uzi61 {{
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    background: rgb(240 242 246);
    box-shadow: 0 5px 8px #6c757d;
}}

.st-emotion-cache-1vt4y43 {{
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    COLOR: WHITE;
    user-select: none;
    background-color: #0461f1;
    border: 1px solid rgba(49, 51, 63, 0.2);
}}

.st-emotion-cache-qcpnpn {{
    border: 1px solid rgb(163, 168, 184);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    background-color: rgb(38, 39, 48);
    MARGIN-TOP: 9PX;
    box-shadow: 0 5px 8px #6c757d;
}}


.st-emotion-cache-15hul6a {{
    user-select: none;
    background-color: #ffc107;
    border: 1px solid rgba(250, 250, 250, 0.2);
    
}}

.st-emotion-cache-1hskohh {{
    margin: 0px;
    padding-right: 2.75rem;
    color: rgb(250, 250, 250);
    border-radius: 0.5rem;
    background: #000;
}}

.st-emotion-cache-12pd2es {{
    margin: 0px;
    padding-right: 2.75rem;
    color: #f0f2f6;
    border-radius: 0.5rem;
    background: #000;
}}

.st-emotion-cache-1r6slb0 {{
    width: calc(33.3333% - 1rem);
    flex: 1 1 calc(33.3333% - 1rem);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}}
.st-emotion-cache-12w0qpk {{
    width: calc(25% - 1rem);
    flex: 1 1 calc(25% - 1rem);
    display: flex;
    flex-direction: row;
    justify-content: CENTER;
    ALIGN-ITEMS: CENTER;
}}



.st-emotion-cache-1kyxreq {{
    display: flex;
    flex-flow: wrap;
    row-gap: 1rem;
    align-items: center;
    justify-content: center;
}}

img {{
    vertical-align: middle;
    border-radius: 10px;
 
}}


    h5 {{
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 600;
    color: rgb(14 14 14);
    padding: 0px 0px 1rem;
    margin: 0px;
    line-height: 1.2;
}}



</style>
"""

# Apply CSS styling to the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)


# Sidebar configuration
with st.sidebar:
    # Display logo image
    st.image("https://raw.githubusercontent.com/Salman7292/Brain-Tumor-Detection-Project/29fdf15114e9a1f81d02441619176acec3f55dbd/Logo4.png", use_column_width=True)

    # Adding a custom style with HTML and CSS for sidebar
    st.markdown("""
        <style>
            .custom-text {
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                color:#ffc107
            }
            .custom-text span {
                color: #04ECF0; /* Color for the word 'Recommendation' */
            }
        </style>
    """, unsafe_allow_html=True)
  

    # Displaying the subheader with custom styling
    st.markdown('<p class="custom-text"> Brain Tumor <span>Detecter</span> App</p>', unsafe_allow_html=True)

    # HTML and CSS for the GitHub button
    github_button_html = """
    <div style="text-align: center; margin-top: 50px;">
        <a class="button" href="https://github.com/Salman7292" target="_blank" rel="noopener noreferrer">Visit my GitHub</a>
    </div>

    <style>
        /* Button styles */
        .button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #ffc107;
            color: black;
            text-decoration: none;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #000345;
            color: white;
            text-decoration: none; /* Remove underline on hover */
        }
    </style>
    """

    # Display the GitHub button in the sidebar
    st.markdown(github_button_html, unsafe_allow_html=True)
    
    # Footer HTML and CSS
    footer_html = """
    <div style="padding:10px; text-align:center;margin-top: 10px;">
        <p style="font-size:20px; color:#ffffff;">Made with ❤️ by Salman Malik</p>
    </div>
    """

    # Display footer in the sidebar
    st.markdown(footer_html, unsafe_allow_html=True)


# Define the option menu for navigation
selections = option_menu(
    menu_title=None,
    options=['Home', "Classify MRI Scan"],
    icons=['house-fill', "file-earmark-medical-fill"],

    menu_icon="cast",
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {
            "padding": "5px 23px",
            "background-color": "#0d6efd",
            "border-radius": "8px",
            "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.25)"
        },
        "icon": {"color": "#f9fafb", "font-size": "18px"},
        "hr": {"color": "#0d6dfdbe"},
        "nav-link": {
            "color": "#f9fafb",
            "font-size": "15px",
            "text-align": "center",
            "margin": "0 10px",
            "--hover-color": "#0761e97e",
            "padding": "10px 10px",
            "border-radius": "16px"
        },
        "nav-link-selected": {"background-color": "#ffc107", "font-size": "12px"},
    }
)

if selections == "Home":
# Define HTML and CSS for the hero section
    code = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Mask Detection App</title>
<style>
    .hero-section {

        padding: 60px 20px;
        text-align: center;
        font-family: Arial, sans-serif;
    }
    .hero-heading {
        font-size: 2.5rem;
        margin-bottom: 20px;
        color: #343a40;
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
    }
    .hero-text {
        font-size: 1.2rem;
        line-height: 1.6;
        color: #6c757d;
        max-width: 800px;
        margin: 0 auto;
    }
</style>
</head>
<body>
<section class="hero-section">
    <div class="container">
        <h1 class="hero-heading">Brain Tumor Detection  Made Simple</h1>
        <p class="hero-text">
            Welcome to our Brain Tumor Detection app. Effortlessly classify MRI images to determine if a brain is healthy or affected by a tumor. Simply upload an MRI scan or provide a URL to get instant and accurate results. Empower your healthcare journey with reliable and precise tumor detection, designed to assist medical professionals and patients alike.
        </p>
    </div>
</section>
</body>
</html>
"""





# Use Streamlit to display the HTML content
    st.markdown(code, unsafe_allow_html=True)

elif selections == "Classify MRI Scan":
    st.markdown(
        """
        <h1 style='text-align: center;'>Insert Your MRI Scan Here</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
        """,
        unsafe_allow_html=True
    )


    Browes_file,Inesrt_from_url=st.tabs(["Browse File","Insert from URL"])

    with Browes_file:
        image = None  # Initialize image variable outside the expander

        with st.expander("Loading Image..."):
            # Image uploader widget
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

            # Check if an image has been uploaded
            if uploaded_image is not None:
                try:
                    # Open the image
                    image = Image.open(uploaded_image)
                    # Resize the image
                    max_size = (400, 400)  # Set the maximum width and height
                    image = image.resize(max_size, Image.Resampling.LANCZOS)
                    st.success("Image uploaded successfully!")

                except Exception as e:
                    st.error(f"An error occurred while uploading the image: {e}")

        # Display the uploaded image outside the expander
        if image is not None:

            st.image(image, caption="Uploaded Image", use_column_width=False)
            classify = st.button("Classify MRI")


            if classify:
                predication=image_preprocessing(uploaded_image)
                st.markdown(f"""
                        <div class="bmiresult">
                            <h3>Info About MRI Scan</h3>
                            <h5>{predication}</h5>

                        </div>
                    """, unsafe_allow_html=True)
                result=generate_tumor_info(predication)

                if result:  
                    st.title(f"Full Detail About {predication}")
                    st.markdown(result.text)
                
        
    with Inesrt_from_url:
        url = st.text_input("Enter the image URL")

        if url:
            try:
                # Fetch the image from the URL
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))

                # Resize the image
                max_size = (400, 400)  # Set the maximum width and height
                image = image.resize(max_size, Image.Resampling.LANCZOS)
                st.success("Image loaded from URL successfully!")

                # Display the image and classification button
                st.image(image, caption="Image from URL", use_column_width=False)
                classify = st.button("Classify MRI",key="url")

                if classify:
                    prediction = image_preprocessing_from_url(url)
                    st.markdown(f"""
                            <div class="bmiresult">
                                <h3>Info About MRI Scan</h3>
                                <h5>{prediction}</h5>
                            </div>
                        """, unsafe_allow_html=True)
                    result=generate_tumor_info(prediction)

                    if result:

                        
                        st.title(f"Full Detail About {prediction}")
                        st.markdown(result.text)


            except Exception as e:
                st.error(f"An error occurred while fetching the image from URL: {e}")



