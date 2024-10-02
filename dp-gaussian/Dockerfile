# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
# COPY requirements.txt .

# Copy the current directory contents into the container at /app
COPY . /app

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container
# COPY dp-gaussian.py .

# CMD ["flask","run"]

# Expose the port your Flask app will run on
EXPOSE 5008

# Define environment variable for Flask
ENV FLASK_APP=dp-gaussian.py 

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5008"]