import os
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to access it

# --- Configuration ---
# IMPORTANT: Your Gemini API key is integrated here.
# It's highly recommended to use environment variables for API keys in production.
# For local development, this will use the provided key if GEMINI_API_KEY env var is not set.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "Your API KEY")

if GEMINI_API_KEY == "Your API KEY":
    print("NOTE: Using the API key directly embedded in app.py. For production, consider using environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# --- Routes ---

@app.route('/')
def serve_index():
    """Serves the index.html file."""
    return send_from_directory('.', 'index.html')

@app.route('/generate', methods=['POST'])
def generate_response():
    """
    Handles requests from the frontend, calls the Gemini API, and returns the response.
    """
    data = request.json
    prompt = data.get('prompt')
    model_name = data.get('model', 'gemini-2.5-flash-preview-05-20') # Default model

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        # Initialize the generative model
        model = genai.GenerativeModel(model_name)

        # Generate content with exponential backoff
        MAX_RETRIES = 5
        INITIAL_DELAY = 1 # seconds

        for i in range(MAX_RETRIES):
            try:
                response = model.generate_content(prompt)
                # Check if the response has text content
                if response.candidates and response.candidates[0].content.parts:
                    return jsonify({'response': response.candidates[0].content.parts[0].text})
                else:
                    # If no text content, it might be an empty generation or specific safety filter
                    print(f"Warning: Model generated no text content for prompt: '{prompt}'")
                    print(f"Full response: {response}")
                    return jsonify({'error': 'Model generated no valid text content. It might be filtered or empty.'}), 500
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
                if i < MAX_RETRIES - 1:
                    delay = INITIAL_DELAY * (2 ** i)
                    time.sleep(delay)
                else:
                    raise # Re-raise the last exception if all retries fail

    except Exception as e:
        print(f"Error generating content: {e}")
        return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the Flask app
    # debug=True allows for auto-reloading on code changes (for development)
    app.run(debug=True, port=5000)

