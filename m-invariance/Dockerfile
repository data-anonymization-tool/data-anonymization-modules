# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app will run on
EXPOSE 5013

# Define environment variable for Flask
ENV FLASK_APP=m-invariance.py

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5013"]