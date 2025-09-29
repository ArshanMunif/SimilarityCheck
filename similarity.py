import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gradio as gr
import nltk
import openai
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return words

def calculate_similarity(job_description, resume):

    job_description_words = preprocess(job_description)
    resume_words = preprocess(resume)

    cv = CountVectorizer()
    count_matrix = cv.fit_transform([" ".join(job_description_words), " ".join(resume_words)])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    matchPercentage = cosine_sim[0][1] * 100
    matchPercentage = round(matchPercentage, 2)  # Simple matching percentage

    # Advanced BERT matching
    model = SentenceTransformer('all-mpnet-base-v2')
    job_description_embedding = model.encode(job_description, convert_to_tensor=True).cpu()
    resume_embedding = model.encode(resume, convert_to_tensor=True).cpu()
    cosine_sim_bert = cosine_similarity([job_description_embedding.numpy()], [resume_embedding.numpy()])
    matchPercentage_bert = cosine_sim_bert[0][0] * 100
    matchPercentage_bert = round(matchPercentage_bert, 2)  # BERT matching percentage

    # return f"Simple Matching: Your CV matches about {matchPercentage}% of the job description.\nAdvanced Matching (BERT): Your CV matches about {matchPercentage_bert}% of the job description."
    return f"Your CV matches about {matchPercentage_bert}% of the job description."

# ui = gr.Interface(
#     fn=calculate_similarity,
#     inputs=[gr.Textbox(label="Job Description", lines = 15), gr.Textbox(label="Resume", lines = 15)],
#     outputs=[gr.Textbox(label="Similarity Scores", lines = 5)],
#     title="Resume-Job Description Similarity Checker",
#     description="Enter a job description and a resume to calculate their similarity scores.",
#     theme= gr.themes.Soft(primary_hue="green"),
#     allow_flagging = "never" # remove the flagging button
# )

# ui.launch(inbrowser=True)

with gr.Blocks(title='CV Check', theme=gr.themes.Soft(primary_hue=gr.themes.colors.indigo, secondary_hue=gr.themes.colors.pink,font=[gr.themes.GoogleFont("Poppins")])) as demo:
    gr.Markdown(
    """
    # Welcome to CV Similarity Checker!
    ### Insert your desired job description and your current CV to see your similarity score!
    """)

    with gr.Row():
        job_description_input = gr.Textbox(label="Job Description", lines = 20)
        resume_input = gr.Textbox(label="Resume", lines = 20)
    with gr.Row():
        calculate_button = gr.Button("Show Similarity")
    output_box = gr.Textbox(label="Similarity Score", lines = 2)


    calculate_button.click(
        fn=calculate_similarity,
        inputs=[job_description_input, resume_input],
        outputs=output_box,
        api_name="calculate_similarity",
    )
    description="Use this app to tailor your resume for specific job descriptions.",
    css="footer {visibility: hidden}"
    
demo.launch(inbrowser=True, share = True)
