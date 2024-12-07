from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash, jsonify
import os
import asyncio
import json
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import google.generativeai as genai
import markdown

# Load environment variables from the .env file, such as API keys and secret keys
load_dotenv()

# Create a Flask application instance
app = Flask(__name__)

# Set the secret key for session management and security; retrieved from environment variables
app.secret_key = os.getenv('SECRET_KEY', 'secret_key')

# -------------------------------------
# Configuration for Tiles Functionality
# -------------------------------------
# These directories and configurations handle the storage of images and tiles
app.config['STATIC_FOLDER'] = 'static'
app.config['OUTPUT_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'output_images')
app.config['PROCESSED_TILES_FOLDER'] = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_tiles')

# Ensure the output directories exist
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_TILES_FOLDER'], exist_ok=True)

# -------------------------------------
# Configuration for Plant Disease Predictor
# -------------------------------------
# Set up upload directory for plant images and define allowed file extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -------------------------------------
# Load the Plant Disease Prediction Model
# -------------------------------------
# Load a pre-trained Keras model for predicting plant diseases
plant_model = tf.keras.models.load_model('final_model.keras')

# Load class indices for decoding prediction results into human-readable class names
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    idx_to_class = {int(k): v for k, v in class_indices.items()}

# -------------------------------------
# Configure Gemini API for Chatbot
# -------------------------------------
# The chatbot uses a generative AI model (Gemini) for responding to user queries
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model2 = genai.GenerativeModel("gemini-1.5-flash")

def allowed_file(filename):
    """
    Check if a given filename has an allowed extension for image uploads.
    Returns True if allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size):
    """
    Load an image from disk, convert to RGB, resize to the target size, 
    and preprocess it into a NumPy array suitable for model prediction.

    Parameters:
        img_path (str): Path to the image file.
        target_size (tuple): Desired (width, height) for the image.

    Returns:
        np.array: Preprocessed image array ready for model inference.
    """
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization to [0,1]
    return img_array

def get_chatbot_response(message, disease):
    """
    Communicate with the Gemini API to get a response from the chatbot.
    The system prompt gives the chatbot context about the plant disease.

    Parameters:
        message (str): The user's message or question.
        disease (str): The predicted disease to provide context to the chatbot.

    Returns:
        str: The chatbot response as HTML.
    """
    system_prompt = (
        f"You are a helpful assistant specializing in plant diseases. "
        f"A user has just been diagnosed with '{disease}'. Provide information, "
        f"care instructions, and answer their queries related to this disease."
    )
    
    # Construct the full prompt to send to the generative model
    full_prompt = f"{system_prompt}\nUser: {message}\nAssistant:"

    try:
        response = model2.generate_content(full_prompt)
        chatbot_reply_markdown = response.text.strip()
        # Convert Markdown to HTML for nicer formatting
        chatbot_reply_html = markdown.markdown(chatbot_reply_markdown)
        return chatbot_reply_html
    except Exception as e:
        print(f"Error communicating with Gemini API: {e}")
        return "I'm sorry, I'm unable to assist right now. Please try again later."

# -------------------------------------
# Flask Routes
# -------------------------------------

# -------------------------------------
# Default Home Route
# -------------------------------------
@app.route('/')
def home():
    # Redirect the user to the Plant Disease Predictor by default
    return redirect(url_for('upload_predict'))

# -------------------------------------
# Tiles Functionality Routes
# -------------------------------------
@app.route('/tiles', methods=['GET', 'POST'])
def tiles():
    """
    Handle the generation and display of zoomed-out tiles from Google Earth images.

    GET: Display the current combined images and tiles if available.
    POST: Accept user input (latitude, longitude, zoom_out, num_tiles), 
          clear old images, and generate new ones by calling the async function.
    """
    if request.method == 'POST':
        # Retrieve user inputs for tile generation
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        zoom_out = int(request.form['zoom_out'])
        num_tiles = int(request.form['num_tiles'])

        # Clear old images from output folders
        for filename in os.listdir(app.config['OUTPUT_FOLDER']):
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        if os.path.exists(app.config['PROCESSED_TILES_FOLDER']):
            for filename in os.listdir(app.config['PROCESSED_TILES_FOLDER']):
                file_path = os.path.join(app.config['PROCESSED_TILES_FOLDER'], filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(app.config['PROCESSED_TILES_FOLDER'], exist_ok=True)

        # Import and call the asynchronous image processing function
        from image_processor import process_zoomed_image
        asyncio.run(process_zoomed_image(
            latitude, 
            longitude, 
            "water_segmentation_unet.keras", 
            output_folder=app.config['OUTPUT_FOLDER'], 
            zoom_out=zoom_out, 
            num_tiles=num_tiles
        ))

    # After processing (or if simply GET), load the available processed images for display
    tile_images = sorted(
        [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if f.startswith('image') and f.endswith('.png')],
        key=lambda x: int(x.replace('image', '').replace('.png', ''))
    )

    processed_tile_images = []
    if os.path.exists(app.config['PROCESSED_TILES_FOLDER']):
        processed_tile_images = sorted(
            [f for f in os.listdir(app.config['PROCESSED_TILES_FOLDER']) if f.startswith('processed_image') and f.endswith('.png')],
            key=lambda x: int(x.replace('processed_image', '').replace('.png', ''))
        )

    # Check if combined images exist
    combined_image = 'combined_image.png' if 'combined_image.png' in os.listdir(app.config['OUTPUT_FOLDER']) else None
    combined_processed_image = 'combined_processed_image.png' if 'combined_processed_image.png' in os.listdir(app.config['OUTPUT_FOLDER']) else None

    # Render the index template with the tile data
    return render_template(
        'index.html',
        tile_images=tile_images,
        processed_tile_images=processed_tile_images,
        combined_image=combined_image,
        combined_processed_image=combined_processed_image
    )

@app.route('/output_images/<filename>')
def output_images(filename):
    """
    Serve individual output (original) images directly from the OUTPUT_FOLDER.
    Used by the template to display generated tiles and combined images.
    """
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/processed_tiles/<filename>')
def processed_tiles(filename):
    """
    Serve individual processed tile images directly from the PROCESSED_TILES_FOLDER.
    """
    return send_from_directory(app.config['PROCESSED_TILES_FOLDER'], filename)

# -------------------------------------
# Plant Disease Predictor Routes
# -------------------------------------
@app.route('/plant', methods=['GET', 'POST'])
def upload_predict():
    """
    Handle uploading of a plant image and predicting the disease.
    GET: Show the upload form.
    POST: Handle file upload, run inference, and display results.
    """
    if request.method == 'POST':
        # Ensure a file has been provided in the request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']

        # Check if the file name is not empty
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Validate the file extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Safely secure the filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            # Preprocess the image and run the model prediction
            img = prepare_image(img_path, target_size=(224, 224))  # Adjust target size as per model requirements
            preds = plant_model.predict(img)
            pred_class = np.argmax(preds, axis=1)[0]
            confidence = preds[0][pred_class]
            class_name = idx_to_class.get(pred_class, "Unknown")

            # Render the result page with prediction details
            return render_template('index2.html',
                                   filename=filename,
                                   prediction=class_name,
                                   confidence=round(float(confidence) * 100, 2))
        else:
            flash('Allowed file types are png, jpg, jpeg, gif')
            return redirect(request.url)

    # For GET requests, just show the upload form page
    return render_template('index2.html')

@app.route('/display/<filename>')
def display_image(filename):
    """
    Redirect to the uploaded image in the static/uploads folder.
    This allows the template to display the uploaded plant image.
    """
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle AJAX requests for the chatbot.
    Expected JSON:
        {
            "message": "<User's message>",
            "disease": "<The predicted disease>"
        }
    Returns a JSON response containing the chatbot's reply.
    """
    data = request.get_json()
    user_message = data.get('message')
    disease = data.get('disease')
    
    # Validate the required parameters
    if not user_message or not disease:
        return jsonify({'reply': 'Invalid request. Please provide both message and disease.'}), 400

    # Get chatbot reply based on the user's message and the predicted disease context
    reply = get_chatbot_response(user_message, disease)
    return jsonify({'reply': reply})

if __name__ == "__main__":
    # Run the Flask development server in debug mode
    app.run(debug=True)
