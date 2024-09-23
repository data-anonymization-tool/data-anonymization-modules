# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY k-anonymity.py .

# # Set the environment variable for Flask
# ENV FLASK_APP=k-anonymity.py
# ENV FLASK_RUN_HOST=0.0.0.0
# ENV FLASK_RUN_PORT=5000

# # Expose the port that the app runs on
# EXPOSE 5000

# Run the Flask app
CMD ["flask", "run"]