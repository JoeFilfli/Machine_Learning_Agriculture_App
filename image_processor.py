# image_processor.py

import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import os
import math
import numpy as np
import tensorflow as tf

def load_model(model_path):
    """
    Load the saved model from the specified path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_model_path = os.path.join(script_dir, model_path)
    print(f"Loading model from {full_model_path}...")
    if not os.path.exists(full_model_path):
        print(f"Model file not found at {full_model_path}")
        raise FileNotFoundError(f"Model file not found at {full_model_path}")
    model = tf.keras.models.load_model(full_model_path)
    print("Model loaded successfully.")
    return model


def preprocess_image(image):
    """
    Preprocess the input image for the model.
    """
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def postprocess_image(image_array, original_size):
    """
    Post-process the model output to convert it back to image format.
    """
    image_array = np.squeeze(image_array)
    image_array = (image_array * 255.0).astype(np.uint8)
    image = Image.fromarray(image_array).resize(original_size)
    return image

async def process_zoomed_image(
    latitude, longitude, model_path, output_folder="static/output_images", zoom_out=4000, num_tiles=168
):
    """
    Captures a Google Earth zoomed-out image, processes it with the model, and saves outputs.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Paths for intermediate and output files
    screenshot_path = os.path.join(output_folder, "zoomed_out_screenshot.png")
    cropped_path = os.path.join(output_folder, "zoomed_out_cropped.png")
    combined_image_path = os.path.join(output_folder, "combined_image.png")
    combined_processed_image_path = os.path.join(output_folder, "combined_processed_image.png")
    processed_tiles_folder = os.path.join(output_folder, "processed_tiles")

    # Ensure the processed tiles folder exists
    os.makedirs(processed_tiles_folder, exist_ok=True)

    # Load the model
    model = load_model(model_path)

    # Step 1: Capture the zoomed-out screenshot using Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Construct the Google Earth URL
        url = f"https://earth.google.com/web/@{latitude},{longitude},{zoom_out}a,35y,0h,0t,0r"
        print(f"Opening URL: {url}")
        await page.goto(url)

        # Wait for the page to load (adjust time if necessary)
        await asyncio.sleep(60)

        # Take the screenshot
        await page.screenshot(path=screenshot_path, full_page=True)
        print(f"Zoomed-out screenshot saved as {screenshot_path}")

        # Close the browser
        await browser.close()

    # Step 2: Crop the screenshot
    image = Image.open(screenshot_path)
    width, height = image.size

    # Remove 924 pixels from the bottom and 182 from the top
    cropped_image = image.crop((0, 182, width, height - 924))
    cropped_image.save(cropped_path)
    print(f"Cropped image saved as {cropped_path}")

    # Step 3: Determine tile size
    cropped_width, cropped_height = cropped_image.size
    total_area = cropped_width * cropped_height
    tile_side = int(math.sqrt(total_area / num_tiles))
    print(f"Tile size: {tile_side}x{tile_side}")

    # Adjust dimensions
    cols = cropped_width // tile_side
    rows = cropped_height // tile_side
    adjusted_width = cols * tile_side
    adjusted_height = rows * tile_side

    cropped_image = cropped_image.crop((0, 0, adjusted_width, adjusted_height))
    print(f"Adjusted cropped image size: {adjusted_width}x{adjusted_height}")

    # Step 4: Split into tiles and process with model
    tile_num = 1
    tile_paths = []
    processed_tile_paths = []

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

            # Process the tile with the model
            input_image = preprocess_image(tile)
            output_image_array = model.predict(input_image)
            output_image = postprocess_image(output_image_array, tile.size)

            processed_tile_filename = f"processed_image{tile_num}.png"
            processed_tile_path = os.path.join(processed_tiles_folder, processed_tile_filename)
            output_image.save(processed_tile_path)
            processed_tile_paths.append(os.path.join('processed_tiles', processed_tile_filename))

            print(f"Processed tile {tile_num} saved as {processed_tile_path}")
            tile_num += 1

    print(f"All tiles and processed tiles saved in {output_folder}")

    # Step 5: Combine original tiles into one image
    combined_image = Image.new('RGB', (adjusted_width, adjusted_height))
    tile_index = 0

    for row in range(rows):
        for col in range(cols):
            tile_path = os.path.join(output_folder, tile_paths[tile_index])
            tile = Image.open(tile_path)
            combined_image.paste(tile, (col * tile_side, row * tile_side))
            tile_index += 1

    combined_image.save(combined_image_path)
    print(f"Combined original image saved as {combined_image_path}")

    # Step 6: Combine processed tiles into one image
    combined_processed_image = Image.new('RGB', (adjusted_width, adjusted_height))
    tile_index = 0

    for row in range(rows):
        for col in range(cols):
            processed_tile_path = os.path.join(output_folder, processed_tile_paths[tile_index])
            tile = Image.open(processed_tile_path)
            combined_processed_image.paste(tile, (col * tile_side, row * tile_side))
            tile_index += 1

    combined_processed_image.save(combined_processed_image_path)
    print(f"Combined processed image saved as {combined_processed_image_path}")

# Function to run the async function from a synchronous context
def generate_images():
    # Example coordinates (modify as needed)
    latitude = 33.6131   # Example latitude
    longitude = 35.7214  # Example longitude
    model_path = "water_segmentation_unet.keras"  # Path to your saved model

    # Run the asynchronous function
    asyncio.run(process_zoomed_image(latitude, longitude, model_path, zoom_out=4000, num_tiles=168))

if __name__ == "__main__":
    generate_images()
