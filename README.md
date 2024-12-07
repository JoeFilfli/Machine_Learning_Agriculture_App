# Agriculture Assistance App

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  - [Plant Disease Detection](#plant-disease-detection)
  - [Farmer Chatbot](#farmer-chatbot)
  - [Water Segmentation](#water-segmentation)
- [Future Extensions](#future-extensions)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Steps](#setup-steps)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Gemini API Integration](#gemini-api-integration)
- [Running the App](#running-the-app)
- [Usage](#usage)
  - [Plant Disease Detection](#plant-disease-detection)
  - [Farmer Chatbot](#farmer-chatbot)
  - [Water Segmentation](#water-segmentation)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
  - [Intersection over Union (IoU)](#intersection-over-union-iou)
- [Contribution](#contribution)
- [License](#license)
- [Contact](#contact)

## Overview

The **Agriculture Assistance App** is a comprehensive tool designed to empower farmers by leveraging advanced machine learning and AI technologies. The app offers:

- **Plant Disease Detection:** Quickly identify diseases affecting various crops through image analysis.
- **Farmer Chatbot:** An intelligent chatbot to provide guidance and solutions for curing detected diseases.
- **Water Segmentation:** Detect and map water bodies in satellite images to aid in water resource management, especially in arid regions.

By integrating these features, the app aims to enhance agricultural productivity, ensure sustainable water usage, and support farmers in making informed decisions.

## Features

### Plant Disease Detection

- **Automatic Identification:** Upload images of plant leaves, and the model will analyze and detect potential diseases.
- **Detailed Insights:** Receive information about the identified disease, including symptoms, causes, and treatment methods.
- **Early Intervention:** Enables farmers to take timely actions to mitigate crop damage.

### Farmer Chatbot

- **Interactive Assistance:** Engage with a conversational AI to seek advice on curing plant diseases.
- **Personalized Recommendations:** Get tailored solutions based on the specific disease detected in your crops.
- **24/7 Availability:** Access support anytime, anywhere, ensuring continuous assistance.

### Water Segmentation

- **Satellite Image Analysis:** Upload satellite images to identify and segment water bodies.
- **Resource Management:** Helps in planning and managing water resources effectively, particularly in desert regions.
- **Environmental Monitoring:** Monitor changes in water availability over time to support sustainable practices.

## Future Extensions

- **Global Water Resource Analysis:** Expand the water segmentation feature to analyze and monitor water resources globally, assessing trends in water availability and potential depletion.
- **Extraterrestrial Water Detection:** Adapt the segmentation algorithms to scan for water on other planets, contributing to space exploration and research.
- **Comprehensive Environmental Analytics:** Integrate additional modules to analyze soil health, crop yield predictions, and climate impact assessments.

## Technologies Used

- **Frontend:** HTML, CSS, JavaScript, Flask Templates
- **Backend:** Python, Flask
- **Machine Learning:** TensorFlow, Keras
- **APIs:** Gemini API for Chatbot Functionality
- **Data Processing:** rsync for data synchronization
- **Visualization:** Matplotlib
- **Version Control:** Git

## Installation

### Prerequisites

- **Python 3.7+**: Ensure Python is installed on your system.
- **pip**: Python package manager.
- **Git**: Version control system.
- **Virtual Environment Tool**: `venv` or `virtualenv` is recommended.

### Setup Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/JoeFilfli/agriculture-assistance-app.git
    cd agriculture-assistance-app
    ```

2. **Create a Virtual Environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Install rsync**

    - **For Linux/Mac:**

        rsync is typically pre-installed. If not, install it using your package manager.

        ```bash
        sudo apt-get install rsync
        ```

    - **For Windows:**

        Install [Cygwin](https://www.cygwin.com/) or use [WSL](https://docs.microsoft.com/en-us/windows/wsl/install) to access rsync.

## Configuration

### Environment Variables

The application requires certain environment variables to be set for proper functioning, especially for integrating the Gemini API and managing model paths.

1. **Create a `.env` File**

    ```bash
    cp .env.example .env
    ```

2. **Edit the `.env` File**

    Open the `.env` file in a text editor and configure the following variables:

    ```env
    # Flask Configuration
    FLASK_APP=app.py
    FLASK_ENV=development
    SECRET_KEY=your_secret_key_here

    # Gemini API Configuration
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

    - **`SECRET_KEY`**: A secret key for Flask sessions. Replace `your_secret_key_here` with a strong, unique key.
    - **`GEMINI_API_KEY`**: Obtain your API key from [Gemini API](https://www.gemini.com/). Replace `your_gemini_api_key_here` with your actual API key.

### Gemini API Integration

Ensure that the Gemini API key is correctly set in the `.env` file as shown above. This key is essential for enabling the chatbot functionality within the app.

## Running the App

1. **Activate the Virtual Environment**

    ```bash
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Run the Flask Application**

    ```bash
    python app.py
    ```

    Alternatively, if using Flask's CLI:

    ```bash
    flask run
    ```

3. **Access the App**

    Open your web browser and navigate to `http://localhost:5000` to access the Agriculture Assistance App.

## Usage

### Plant Disease Detection

1. **Navigate to Disease Detection**

    From the homepage, select the "Plant Disease Detection" feature.

2. **Upload Plant Image**

    Upload a clear image of the plant or its leaves.

3. **View Results**

    The model will analyze the image and display any detected diseases along with information on symptoms, causes, and treatment methods.

### Farmer Chatbot

1. **Access the Chatbot**

    From the homepage or navigation menu, select the "Farmer Chatbot" feature.

2. **Interact with the Chatbot**

    Engage in a conversation by typing queries related to plant diseases, treatment methods, or general farming advice.

3. **Receive Guidance**

    The chatbot, powered by the Gemini API, will provide tailored responses to assist in curing diseases and improving crop health.

### Water Segmentation

1. **Navigate to Water Segmentation**

    From the homepage, select the "Water Segmentation" feature.

2. **Input Geographical Coordinates**

    - **Latitude and Longitude:** Enter the specific latitude and longitude coordinates of the area you wish to analyze.
    - **Zoom Level:** Adjust the zoom level to obtain a broader or more detailed view of the satellite image.

3. **Analyze Water Bodies**

    The model will identify and segment water bodies, providing a visual map and relevant data for effective water resource management.

## Model Training

The repository includes pre-trained models for both plant disease detection and water segmentation. However, if you wish to train or fine-tune these models:

1. **Navigate to Model Training Directory**

    ```bash
    cd Model_Training
    ```

2. **Review Training Scripts**

    - **`image_processor.py`**: Handles data preprocessing and augmentation.
    - **`final_model.keras`**: Pre-trained plant disease detection model.
    - **`water_segmentation_unet.keras`**: Pre-trained water segmentation U-Net model.

3. **Modify Training Parameters**

    Adjust hyperparameters, dataset paths, and other configurations as needed within the training scripts.

4. **Execute Training Scripts**

    ```bash
    python train_disease_model.py
    python train_water_segmentation_model.py
    ```

    Replace `train_disease_model.py` and `train_water_segmentation_model.py` with the actual script names used for training.

5. **Save Trained Models**

    After training, ensure that the models are saved to the designated paths and update the `.env` file accordingly.

## Contribution

Contributions are welcome! Please follow these steps to contribute to the project:

1. **Fork the Repository**

2. **Create a New Branch**

    ```bash
    git checkout -b feature/YourFeatureName
    ```

3. **Commit Your Changes**

    ```bash
    git commit -m "Add your message here"
    ```

4. **Push to the Branch**

    ```bash
    git push origin feature/YourFeatureName
    ```

5. **Open a Pull Request**

Please ensure your contributions adhere to the project's coding standards and include appropriate tests and documentation.

## Contact

For any inquiries or support, please contact:

- **Name:** Joe Filfli
- **Email:** joefilfli48@gmail.com
- **GitHub:** [@JoeFilfli](https://github.com/JoeFilfli)

---

*Thank you for using the Agriculture Assistance App! We are committed to supporting farmers with cutting-edge technology to ensure sustainable and productive agricultural practices.*
