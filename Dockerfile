# Use Python 3.12-slim image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Install required system packages for Playwright and TensorFlow
RUN apt-get update && apt-get install -y \
    wget \
    gpg \
    libnss3 \
    libxss1 \
    libasound2 \
    libatk1.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    build-essential \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and its dependencies
RUN pip install playwright && playwright install

# Copy the rest of the application code
COPY . .

# Set TensorFlow environment variables to suppress warnings
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
# Set Flask environment to production
ENV FLASK_ENV=production

# Expose application port
EXPOSE 5000

# Copy the .env file
COPY .env /app/.env

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Command to run the application
CMD ["python", "app.py"]

