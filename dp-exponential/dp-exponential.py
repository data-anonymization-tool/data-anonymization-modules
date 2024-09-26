import pandas as pd
import numpy as np
import sys
from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Define upload and anonymized file directories
UPLOAD_FOLDER = os.path.abspath('./uploads')
ANONYMIZED_FOLDER = os.path.abspath('./anonymized')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANONYMIZED_FOLDER'] = ANONYMIZED_FOLDER

# Create directories if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(ANONYMIZED_FOLDER):
    os.makedirs(ANONYMIZED_FOLDER)

def exponential_mechanism(choices, utility_func, sensitivity, epsilon):
    scores = np.array([utility_func(choice) for choice in choices])
    probabilities = np.exp(epsilon * scores / (2 * sensitivity))
    probabilities /= probabilities.sum()  # Normalize to create a probability distribution
    selected_choice = np.random.choice(choices, p=probabilities)
    return selected_choice

def apply_exponential_mechanism(df, column, sensitivity, epsilon):
    print(f"Applying Exponential mechanism to the categorical column '{column}'")
    unique_values = df[column].unique()
    
    def utility_func(value):
        return df[column].value_counts().get(value, 0)
    
    df[column] = df[column].apply(lambda x: exponential_mechanism(unique_values, utility_func, sensitivity, epsilon))
    return df

@app.route('/dp-exponential', methods=['POST'])
def apply_exponential():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    
    # Get the parameters from the request
        column = request.form.get('column')
        direct_identifiers = request.form.get('direct_identifiers', '')
        sensitivity = float(request.form.get('sensitivity', 1))
        epsilon = float(request.form.get('epsilon', 1.0))

        # Load the dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Split the comma-separated list of direct identifiers into a list
        direct_identifiers_list = [col.strip() for col in direct_identifiers.split(',')] if direct_identifiers else []

        # Check if the column exists in the DataFrame
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found in the file."}), 400

        # Remove direct identifier columns if specified
        if direct_identifiers_list:
            df.drop(columns=direct_identifiers_list, errors='ignore', inplace=True)

        if pd.api.types.is_numeric_dtype(df[column]):
            print(f"Error: Column '{column}' is numerical. Please use the Laplace or Gaussian mechanism for numerical data.")
            sys.exit(1)

        df = apply_exponential_mechanism(df, column, sensitivity, epsilon)

        # Save the anonymized dataset to a file
        anonymized_filename = f"anonymized_{uuid.uuid4().hex}_{filename}"
        anonymized_filepath = os.path.join(app.config['ANONYMIZED_FOLDER'], anonymized_filename)
        df.to_csv(anonymized_filepath, index=False) if file_path.endswith('.csv') else df.to_excel(anonymized_filepath, index=False)

        # Send the anonymized file as a response
        return send_file(anonymized_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
