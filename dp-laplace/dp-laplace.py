from flask import Flask, request, jsonify, send_file
import pandas as pd
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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

def laplace_mechanism(value, sensitivity, epsilon):
    import numpy as np
    noise = np.random.laplace(0, sensitivity / epsilon)
    return round(value + noise) if isinstance(value, int) else value + noise

def apply_laplace_mechanism(df, column, sensitivity, epsilon):
    df[column] = df[column].apply(lambda x: laplace_mechanism(x, sensitivity, epsilon))
    return df

@app.route('/dp-laplace/metadata', methods=['GET'])
def get_metadata():
    return send_file('dp-laplace.json', as_attachment=False)

@app.route('/dp-laplace/', methods=['POST'])
def apply_laplace():
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
        column = request.form.get('Column to be anonymized')
        direct_identifiers = request.form.get('Direct Identifier Columns', '')
        sensitivity = float(request.form.get('Sensitivity', 1))
        epsilon = float(request.form.get('Epsilon', 1.0))

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

        # Apply Laplace mechanism to the specified column
        if not pd.api.types.is_numeric_dtype(df[column]):
            return jsonify({"error": f"Column '{column}' is not numerical. Please use exponential mechanism for categorical data."}), 400

        df = apply_laplace_mechanism(df, column, sensitivity, epsilon)

        # Save the anonymized dataset to a file
         # Generate timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        anonymized_filename = f"anonymized_{timestamp}_{filename}"
        anonymized_filepath = os.path.join(app.config['ANONYMIZED_FOLDER'], anonymized_filename)
        df.to_csv(anonymized_filepath, index=False) if file_path.endswith('.csv') else df.to_excel(anonymized_filepath, index=False)

        # Send the anonymized file as a response
        return send_file(anonymized_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
