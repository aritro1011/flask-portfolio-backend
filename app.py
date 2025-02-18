from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS  # Import CORS to enable cross-origin requests
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API with the key
genai.configure(api_key=api_key)


# Hardcoded resume data (for context)
resume_data = """
ARITRO SEN
Data Scientist | Machine Learning Engineer
Languages Spoken: English, Hindi, Bengali, Marathi
DOB: 10th November 2003
Age: 21
Contact Information
Email: aritro1011@gmail.com
Phone: +91 9022314977
Location: New Delhi
GitHub: GitHub Profile
LinkedIn: LinkedIn Profile
Portfolio: Portfolio Link
Education
Vellore Institute of Technology, Bhopal
Bachelor of Technology, Computer Science with Specialization in AI/ML

CGPA: 8.08 (2021–2025)
Bharatiya Vidya Bhavans, Nagpur
Class 12th

Percentage: 81.4% (2021)
Class 10th
Percentage: 91.6% (2019)
Skills
Languages:

Python, SQL
Data Analysis:

Exploratory Data Analysis (EDA), Trend Analysis, MS Excel, Data Visualization Tools
Machine Learning:

Machine Learning Algorithms, Deep Learning, NLP, OpenCV, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Vector Databases (FAISS, ChromaDB, Pinecone)
Frameworks/Libraries:

Data Libraries: Pandas, NumPy, Matplotlib
ML/DL Frameworks: TensorFlow, PyTorch, Huggingface Transformers, LangChain
Development Frameworks: Streamlit, Flask
Tools/Platforms:

IntelliJ, Google Colab, MySQL Workbench, VS Code, Git/GitHub
Experience
Data and Finance Analytics Intern
Finlatics (Oct 2024 – Dec 2024, Remote)

Conducted comprehensive Exploratory Data Analysis (EDA) on large financial datasets, uncovering actionable insights.
Preprocessed raw financial data, including cleaning, handling missing values, and transforming it for analysis.
Collaborated with cross-functional teams for a capstone project demonstrating advanced EDA, data preprocessing, and visualization techniques.
Academic Projects
AskDoc RAG

Technologies: Python, LangChain, Gemini API, RAG, Streamlit
Developed a RAG framework, improving answer relevance by 20%.
Integrated ChromaDB for efficient vector storage and document retrieval.
Built a Streamlit interface allowing users to upload documents and receive contextually enriched answers.
Gemini-Based Curriculum Generator

Technologies: Python, Streamlit, Gemini LLM
Created an AI-powered curriculum generator using Gemini, improving content relevance by 15%.
Integrated NLP techniques to generate structured learning paths.
Applied prompt engineering to structure outputs and reference research papers.
Rubik’s Cube Simulator with Gesture Controls

Technologies: OpenCV, MediaPipe, OpenGL
Developed an interactive Rubik’s Cube simulator with gesture controls.
Improved hand gesture recognition accuracy by 15% using OpenCV and MediaPipe.
LLMs TicTacToe

Technologies: Gemini AI, Llama 3 AI
Interactive Tic-Tac-Toe game with AI players using Gemini AI and Llama 3 AI.
Recorded detailed game results (moves, temperatures, outcomes) in a CSV file for analysis.
CNN-Based Human Emotion Detection Chrome Extension

Developed a Chrome extension using CNN for real-time human emotion detection.
Certifications
GenAI for Data-Driven Business Decision-Making – IIM Mumbai
Certified Cloud Practitioner – Amazon Web Services
Privacy and Security in Online Social Media – Swayam NPTEL
Computer Vision – VITyarthi (VIT Bhopal)
Co-curricular Activities
1st Runner-Up, AI Club's Ideathon – AI Conclave 2022

AI Club, VIT Bhopal
Secured 1st Runner-Up position out of 50+ teams.
Awarded a certificate and a cash prize of Rs. 6,000.
Member, AI Club VIT Bhopal

2022–2023, Bhopal, India
Led the "AI Crypt" event, attracting 100+ participants and securing Rs. 30,000 in sponsorships.

"""

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes, allowing requests from the frontend
CORS(app)  # You can also restrict to specific origins: CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# Function to chat with the bot (Generates answers based on resume data)
def chat(question):
    prompt = f"""Resume Data:\n{resume_data}\n\nQuestion: {question}\n\nAnswer the user's question based solely on the context provided in the resume. 

- Rephrase the information from the resume in your own words; do not copy text verbatim.
-Answer in a professional and informative manner.Do not provide unstructured responses
- when answering in a large paragraph, use line breaks.
- If the information is not available in the resume, provide the most relevant information that can be inferred or summarized from the resume.
- Do not create information about the candidate that is not present in the resume.
- Present the answer in a clear, concise, and professional manner.
- Use a structured bulleted format to present the answer with key points.
- Avoid providing unnecessary or irrelevant details.
- Do not attempt to answer questions that cannot be answered with the provided resume data.
- Respond as if you are Aritro Sen, not as Gemini.
"""

    # Generate the response using Gemini
    gemini_model = genai.GenerativeModel("gemini-pro")
    response = gemini_model.generate_content(prompt)

    # Return the generated answer
    return response.text.strip()

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400  # Handle case where no question is provided
    
    try:
        # Get the response from the chatbot
        response = chat(question)
        return jsonify({"answer": response})
    
    except Exception as e:
        # Handle any exceptions that occur during the chat process
        print(f"Error: {e}")
        return jsonify({"error": "There was an issue processing your request. Please try again later."}), 500

if __name__ == '__main__':
    app.run(debug=True)
