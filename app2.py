import os
import json
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import markdown

# Load environment variables from .env file
load_dotenv()

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.getenv('SECRET_KEY', 'secret_key')  

# Load the Keras model
model = load_model('final_model.keras')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    idx_to_class = {int(k): v for k, v in class_indices.items()}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size):
    img = Image.open(img_path)
    img = img.convert('RGB')  # Ensure image is in RGB format
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize as per training
    return img_array

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model2 = genai.GenerativeModel("gemini-1.5-flash")
def get_chatbot_response(message, disease):
    """
    Sends a message to the Gemini API and returns the response.
    
    Parameters:
    - message (str): User's message to the chatbot.
    - disease (str): Predicted disease to provide context.
    
    Returns:
    - response (str): Chatbot's reply.
    """
    system_prompt = (
        f"You are a helpful assistant specializing in plant diseases. "
        f"A user has just been diagnosed with '{disease}'. Provide information, care instructions, and answer their queries related to this disease."
    )
    
    # Combine system prompt and user message
    full_prompt = f"{system_prompt}\nUser: {message}\nAssistant:"
    
    try:
        response = model2.generate_content(
            full_prompt,    
        )
        chatbot_reply_markdown = response.text.strip()
        
        # Convert Markdown to HTML
        chatbot_reply_html = markdown.markdown(chatbot_reply_markdown)
        
        return chatbot_reply_html
    except Exception as e:
        print(f"Error communicating with Gemini API: {e}")
        return "I'm sorry, I'm unable to assist right now. Please try again later."

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser may submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            
            # Prepare image
            img = prepare_image(img_path, target_size=(224, 224))  # Adjust target_size as per your model
            
            # Predict
            preds = model.predict(img)
            pred_class = np.argmax(preds, axis=1)[0]
            confidence = preds[0][pred_class]
            class_name = idx_to_class.get(pred_class, "Unknown")
            
            return render_template('index2.html',
                                   filename=filename,
                                   prediction=class_name,
                                   confidence=round(float(confidence) * 100, 2))
        else:
            flash('Allowed file types are png, jpg, jpeg, gif')
            return redirect(request.url)
    return render_template('index2.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chatbot interactions.
    
    Expects JSON data with:
    - message: The user's message to the chatbot.
    - disease: The predicted disease for context.
    
    Returns:
    - JSON response with the chatbot's reply.
    """
    data = request.get_json()
    user_message = data.get('message')
    disease = data.get('disease')
    
    if not user_message or not disease:
        return jsonify({'reply': 'Invalid request. Please provide both message and disease.'}), 400
    
    # Get chatbot response
    reply = get_chatbot_response(user_message, disease)
    
    return jsonify({'reply': reply})

if __name__ == "__main__":
    app.run(debug=True)
