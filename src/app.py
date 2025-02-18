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
resume_data = """
ARITRO SEN
Data Scientist | Machine Learning Engineer
Languages Spoken: English, Hindi, Bengali, Marathi
DOB: 10th November 2003
Age: 21
...
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
