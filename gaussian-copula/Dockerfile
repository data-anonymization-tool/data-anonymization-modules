# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
# COPY requirements.txt .

# Copy the current directory contents into the container at /app
COPY . /app

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#  Copy the rest of the application code into the container
# COPY gaussian-copula.py .

# Run the Flask app
# CMD ["flask", "run"]

# Expose the port your Flask app will run on
EXPOSE 5004

# Define environment variable for Flask
ENV FLASK_APP=gaussian-copula.py

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5004"]
