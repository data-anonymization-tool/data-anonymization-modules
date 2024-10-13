# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# COPY requirements.txt .

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# COPY ctgan-synthesis.py .

# Run the Flask app
# CMD ["flask", "run"]

# Expose the port your Flask app will run on
EXPOSE 5005

# Define environment variable for Flask
ENV FLASK_APP=ctgan-synthesis.py

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5005"]