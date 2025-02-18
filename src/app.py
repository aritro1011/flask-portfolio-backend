from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS  # Import CORS to enable cross-origin requests
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API with the key
genai.configure(api_key=api_key)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS only for the frontend domain
CORS(app, resources={r"/api/*": {"origins": "https://aritro-portfolio.vercel.app"}})

# Hardcoded resume data (for context)
resume_data ="""
ARITRO SEN
--------------------------------------------
Data Scientist | Machine Learning Engineer
Languages Spoken: English, Hindi, Bengali, Marathi
DOB: 10th November 2003
Age: 21
Contact Information
Email: aritro1011@gmail.com
Phone: +91 9022314977
Location: New Delhi
----------------------------------------------
Education
Vellore Institute of Technology, Bhopal
Bachelor of Technology, Computer Science with Specialization in AI/ML

CGPA: 8.08 (2021–2025)
Bharatiya Vidya Bhavans, Nagpur
Class 12th

Percentage: 81.4% (2021)
Class 10th
Percentage: 91.6% (2019)
-------------------------------------------------
Objective
Passionate and driven Data Scientist and Machine Learning Engineer with a strong background in AI, machine learning, and data analytics. 
I am eager to apply my technical skills and problem-solving abilities in a dynamic environment to drive data-driven decisions and innovations. 
Seeking an opportunity to contribute to impactful projects while continuing to grow in the field of artificial intelligence and machine learning.
-------------------------------------------------
Skills
-Languages:
Python, SQL

-Data Analysis:
Exploratory Data Analysis (EDA), Trend Analysis, MS Excel, Data Visualization Tools

-Machine Learning & AI:
Supervised & Unsupervised Learning, Deep Learning, NLP, OpenCV, Large Language Models (LLMs), 
Retrieval-Augmented Generation (RAG), Vector Databases (FAISS, ChromaDB, Pinecone)

-Frameworks/Libraries:
Data Libraries: Pandas, NumPy, Matplotlib, Seaborn
ML/DL Frameworks: TensorFlow, PyTorch, Huggingface Transformers, LangChain
Development Frameworks: Streamlit, Flask, FastAPI

-Tools/Platforms:
IntelliJ, Google Colab, MySQL Workbench, VS Code, Git/GitHub
----------------------------------------------------------------------------
Experience
Data and Finance Analytics Intern
Finlatics (Oct 2024 – Dec 2024, Remote)

Conducted comprehensive Exploratory Data Analysis (EDA) on large financial datasets, uncovering actionable insights.
Preprocessed raw financial data, including cleaning, handling missing values, and transforming it for analysis.
Collaborated with cross-functional teams for a capstone project demonstrating advanced EDA, data preprocessing, and visualization techniques.
----------------------------------------------------------------------------
Academic Projects
-AskDoc RAG
Technologies: Python, LangChain, Gemini API, RAG, Streamlit
Developed a RAG framework, improving answer relevance by 20%.
Integrated ChromaDB for efficient vector storage and document retrieval.
Built a Streamlit interface allowing users to upload documents and receive contextually enriched answers.

-Gemini-Based Curriculum Generator
Technologies: Python, Streamlit, Gemini LLM
         -Created an AI-powered curriculum generator using Gemini, improving content relevance by 15%.
         -Integrated NLP techniques to generate structured learning paths.
         -Applied prompt engineering to structure outputs and reference research papers.
         
-Rubik’s Cube Simulator with Gesture Controls
Technologies: OpenCV, MediaPipe, OpenGL
        -Developed an interactive Rubik’s Cube simulator with gesture controls, allowing users to manipulate the cube in a 3D environment.
        -Improved hand gesture recognition accuracy by 15% using OpenCV and MediaPipe, enhancing the user experience.
        -Integrated OpenGL for rendering real-time 3D graphics, delivering a smooth visual experience.


-LLMs TicTacToe
Technologies: Gemini AI, Llama 3.3 70B AI
        -Developed an interactive Tic-Tac-Toe game featuring AI players powered by Gemini and Llama 3 AI.
        -Recorded detailed game results, including moves, temperatures, and outcomes, to analyze player behavior and game strategies.
        -Implemented AI decision-making algorithms to simulate intelligent opponents, improving the game’s challenge level.

-CNN-Based Human Emotion Detection Chrome Extension
Technologies: Python, TensorFlow, OpenCV, JavaScript
        
        -Developed a Model using Convolutional Neural Networks (CNN) for real-time human emotion detection through webcam input.
        -Trained the model on various facial expression datasets, achieving an accuracy rate of 85% for detecting emotions.
        -The model can be integrated into a web platform to analyze user emotions during online meetings, providing real-time feedback.
--------------------------------------------------------------------------------------
Certifications
        -GenAI for Data-Driven Business Decision-Making – IIM Mumbai
        -Certified Cloud Practitioner – Amazon Web Services
        -Privacy and Security in Online Social Media – Swayam NPTEL
        -Computer Vision – VITyarthi (VIT Bhopal)
--------------------------------------------------------------------------------------
Co-curricular Activities
-1st Runner-Up, AI Club's Ideathon – AI Conclave 2022
AI Club, VIT Bhopal
Secured 1st Runner-Up position out of 50+ teams.
Awarded a certificate and a cash prize of Rs. 6,000.

-Member, AI Club VIT Bhopal
2022–2023, Bhopal, India
Led the "AI Crypt" event, attracting 100+ participants and securing Rs. 30,000 in sponsorships.
---------------------------------------------------------------------------------------
Soft Skills
    -Problem-Solving: Ability to break down complex problems into manageable parts and solve them using structured approaches.
    -Communication: Strong ability to communicate technical concepts to non-technical stakeholders, ensuring that project goals and results are clearly understood.
    -Teamwork: Collaborated effectively with cross-functional teams in both academic and professional environments to achieve project objectives.
    -Leadership: Led teams during academic projects, motivating team members, and ensuring that deadlines were met while maintaining a high standard of work.
---------------------------------------------------------------------------------------

"""

# Function to chat with the bot (Generates answers based on resume data)
def chat(question):
    prompt = f"""Resume Data:\n{resume_data}\n\nQuestion: {question}\n\nAnswer the user's question based solely on the context provided in the resume. 

- Rephrase the information from the resume in your own words; do not copy text verbatim.
- Answer in a professional and informative manner.
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
