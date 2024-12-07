import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import os
import math
import numpy as np
import tensorflow as tf

def load_model(model_path):
    """
    Load a TensorFlow Keras model from the given model_path.

    Parameters:
        model_path (str): Relative path to the saved Keras model file.

    Returns:
        tf.keras.Model: The loaded model ready for inference.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_model_path = os.path.join(script_dir, model_path)
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found at {full_model_path}")
    model = tf.keras.models.load_model(full_model_path)
    return model

def preprocess_image(image):
    """
    Resize and normalize the image for prediction.

    Parameters:
        image (PIL.Image): A PIL Image object.

    Returns:
        np.array: A batch of one image, resized to (256,256) and normalized.
    """
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)

def postprocess_image(image_array, original_size):
    """
    Convert model output back into a PIL image, resizing it to the original tile size.

    Parameters:
        image_array (np.array): Model prediction output array.
        original_size (tuple): The original size (width, height) of the tile before resizing.

    Returns:
        PIL.Image: The post-processed image.
    """
    image_array = np.squeeze(image_array)
    image_array = (image_array * 255.0).astype(np.uint8)
    image = Image.fromarray(image_array).resize(original_size)
    return image

async def process_zoomed_image(latitude, longitude, model_path, output_folder="static/output_images", zoom_out=4000, num_tiles=168):
    """
    Asynchronously fetch a screenshot from Google Earth using Playwright, process the image by splitting 
    it into tiles, running a model prediction on each tile, and then combine them into original and processed versions.

    Parameters:
        latitude (float): Latitude of the location on Google Earth.
        longitude (float): Longitude of the location on Google Earth.
        model_path (str): Path to the model file for tile processing.
        output_folder (str): The directory where output images will be saved.
        zoom_out (int): The altitude/zoom level parameter for Google Earth.
        num_tiles (int): Number of tiles to split the image into.

    Steps:
        1. Launch a headless browser and navigate to Google Earth at the specified coordinates and zoom.
        2. Take a full-page screenshot.
        3. Crop the screenshot to remove unwanted UI elements.
        4. Calculate tile sizes based on the total area and the desired number of tiles.
        5. Split the cropped image into tiles and use the loaded model to process each tile.
        6. Save both original and processed tiles.
        7. Combine all original tiles into one image, and all processed tiles into another image.
        8. Save the combined images for display in the Flask app.
    """

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define file paths for intermediate and output images
    screenshot_path = os.path.join(output_folder, "zoomed_out_screenshot.png")
    cropped_path = os.path.join(output_folder, "zoomed_out_cropped.png")
    combined_image_path = os.path.join(output_folder, "combined_image.png")
    combined_processed_image_path = os.path.join(output_folder, "combined_processed_image.png")
    processed_tiles_folder = os.path.join(output_folder, "processed_tiles")
    os.makedirs(processed_tiles_folder, exist_ok=True)

    # Load the model used for processing the tiles
    model = load_model(model_path)

    # Capture the screenshot of the given location on Google Earth
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        url = f"https://earth.google.com/web/@{latitude},{longitude},{zoom_out}a,35y,0h,0t,0r"
        await page.goto(url)
        await asyncio.sleep(60)  # Wait sufficiently for page and imagery to load
        await page.screenshot(path=screenshot_path, full_page=True)
        await browser.close()

    # Open and crop the screenshot to remove unneeded parts (e.g., UI elements)
    image = Image.open(screenshot_path)
    width, height = image.size
    cropped_image = image.crop((0, 182, width, height - 924))
    cropped_image.save(cropped_path)

    # Determine tile dimensions
    cropped_width, cropped_height = cropped_image.size
    total_area = cropped_width * cropped_height
    tile_side = int(math.sqrt(total_area / num_tiles))

    cols = cropped_width // tile_side
    rows = cropped_height // tile_side
    adjusted_width = cols * tile_side
    adjusted_height = rows * tile_side

    # Further crop image to fit exactly into the calculated tile dimensions
    cropped_image = cropped_image.crop((0, 0, adjusted_width, adjusted_height))

    # Initialize counters and lists to keep track of tile paths
    tile_num = 1
    tile_paths = []
    processed_tile_paths = []

    # Split the image into tiles and process each tile with the model
    for row in range(rows):
        for col in range(cols):
            left = col * tile_side
            top = row * tile_side
            right = left + tile_side
            bottom = top + tile_side

            tile = cropped_image.crop((left, top, right, bottom))
            tile_filename = f"image{tile_num}.png"
            tile_path = os.path.join(output_folder, tile_filename)
            tile.save(tile_path)
            tile_paths.append(tile_filename)

            # Run model prediction on the tile
            input_image = preprocess_image(tile)
            output_image_array = model.predict(input_image)
            output_image = postprocess_image(output_image_array, tile.size)

            # Save the processed tile
            processed_tile_filename = f"processed_image{tile_num}.png"
            processed_tile_path = os.path.join(processed_tiles_folder, processed_tile_filename)
            output_image.save(processed_tile_path)
            processed_tile_paths.append(os.path.join('processed_tiles', processed_tile_filename))

            tile_num += 1

    # Combine original tiles into a single large image
    combined_image = Image.new('RGB', (adjusted_width, adjusted_height))
    tile_index = 0
    for row in range(rows):
        for col in range(cols):
            tile_path = os.path.join(output_folder, tile_paths[tile_index])
            tile = Image.open(tile_path)
            combined_image.paste(tile, (col * tile_side, row * tile_side))
            tile_index += 1
    combined_image.save(combined_image_path)

    # Combine processed tiles into a single large image
    combined_processed_image = Image.new('RGB', (adjusted_width, adjusted_height))
    tile_index = 0
    for row in range(rows):
        for col in range(cols):
            processed_tile_path = os.path.join(output_folder, processed_tile_paths[tile_index])
            tile = Image.open(processed_tile_path)
            combined_processed_image.paste(tile, (col * tile_side, row * tile_side))
            tile_index += 1
    combined_processed_image.save(combined_processed_image_path)
