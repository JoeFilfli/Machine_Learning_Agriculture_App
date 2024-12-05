# app.py

from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Configuration
app.config['STATIC_FOLDER'] = 'static'
app.config['OUTPUT_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'output_images')
app.config['PROCESSED_TILES_FOLDER'] = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_tiles')

@app.route('/')
def index():
    # Get list of original tile images
    tile_images = sorted(
        [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if f.startswith('image') and f.endswith('.png')],
        key=lambda x: int(x.replace('image', '').replace('.png', ''))
    )

    # Get list of processed tile images
    processed_tile_images = sorted(
        [f for f in os.listdir(app.config['PROCESSED_TILES_FOLDER']) if f.startswith('processed_image') and f.endswith('.png')],
        key=lambda x: int(x.replace('processed_image', '').replace('.png', ''))
    )

    # Get combined images
    combined_image = 'combined_image.png' if 'combined_image.png' in os.listdir(app.config['OUTPUT_FOLDER']) else None
    combined_processed_image = 'combined_processed_image.png' if 'combined_processed_image.png' in os.listdir(app.config['OUTPUT_FOLDER']) else None

    return render_template(
        'index.html',
        tile_images=tile_images,
        processed_tile_images=processed_tile_images,
        combined_image=combined_image,
        combined_processed_image=combined_processed_image
    )

@app.route('/output_images/<filename>')
def output_images(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/processed_tiles/<filename>')
def processed_tiles(filename):
    return send_from_directory(app.config['PROCESSED_TILES_FOLDER'], filename)

if __name__ == '__main__':
    # Before starting the app, generate images if they don't exist
    output_folder = app.config['OUTPUT_FOLDER']
    processed_tiles_folder = app.config['PROCESSED_TILES_FOLDER']

    if not os.path.exists(output_folder) or not os.listdir(output_folder):
        from image_processor import generate_images
        generate_images()
    elif not os.path.exists(processed_tiles_folder) or not os.listdir(processed_tiles_folder):
        from image_processor import generate_images
        generate_images()

    app.run(debug=True)

