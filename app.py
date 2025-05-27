from flask import Flask, request, jsonify
import httpx
import openai
import os
import requests
import pdfplumber
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app) 

api_key = os.environ.get("OPENAI_API_KEY")

http_client = httpx.Client(
  timeout=120.0,
  limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
  http2=False 
)

client = openai.OpenAI(
  api_key=api_key,
  http_client=http_client
)

@app.route('/ai-analyzer', methods=['POST'])
def ai_analyzer():
    """
      Analyze a candidate's CV against job requirements.
      ---
      parameters:
        - name: pdf_url
          in: body
          type: string
          required: true
          description: The pre-signed URL to the PDF file of the candidate's CV.
        - name: job
          in: body
          required: true
          type: string
          description: The job requirements for the position.
      responses:
        200:
          description: Analysis result from OpenAI.
          schema:
            type: object
            properties:
              response:
                type: object
                properties:
                  score:
                    type: integer
                    description: Score from 0 to 100 indicating the candidate's suitability.
                  assessment:
                    type: string
                    description: Verbal assessment of the candidate's abilities.
        400:
          description: Failed to download PDF file.
      """
    data = request.json
    pdf_url = data.get('pdf_url')
    print('pdf_url:', pdf_url)
    if not pdf_url:
        return jsonify({"error": "pdf_url is required"}), 400
    job_requirements = data.get('job')
    print('job_requirements:', job_requirements)
    try:
      response = requests.get(pdf_url, verify=False)  # Disable SSL verification for testing purposes
      response.raise_for_status()
    except requests.exceptions.SSLError:
        return jsonify({"error": "SSL error occurred. Please provide a valid HTTPS URL."}), 400
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch the PDF file: {str(e)}"}), 400

    pdf_file_path = 'temp.pdf'
    with open(pdf_file_path, 'wb') as f:
        f.write(response.content)
 
    pdf_text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text() + "\n"
    os.remove(pdf_file_path)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI model that analyzes CVs for job suitability."
            },
            {
                "role": "user",
                "content": f"""You have been provided with a job description and the content of a candidate's CV.
                Your task is to evaluate the candidate's suitability for the position based on their experience, knowledge, education, and personal fit with the job requirements.
                Please provide a score from 0 to 100 and a verbal assessment of the candidate's abilities and suitability for the position.
                The response should be in JSON format (without the word JSON itself):
                {{
                    "score": x,
                    "assessment": "..."
                }} 
                Job requirements: {job_requirements},
                The CV content: {pdf_text}
                """
            }
        ]
    )

    answer = response.choices[0].message.content
    return jsonify({"response": answer})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

