from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
import json

app = Flask(__name__)

# Ensure the temp directory exists
TEMP_DIR = "temp_tvae"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.route('/tvae/metadata', methods=['GET'])
def get_metadata():
    return send_file('tvae.json', as_attachment=False)

# Route for uploading the CSV file and generating synthetic data
@app.route("/tvae", methods=["POST"])
def data_synthesis_tvae():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    columns_to_anonymize = request.form.get("columns_to_anonymize")
    if columns_to_anonymize:
        columns_to_anonymize = json.loads(columns_to_anonymize)
    else:
        columns_to_anonymize = []

    identifying_attributes = request.form.get("identifying_attributes")
    if identifying_attributes:
        identifying_attributes = json.loads(identifying_attributes)
    else:
        identifying_attributes = []

    try:
        # Save the file to a temporary location
        file_path = os.path.join(TEMP_DIR, file.filename)
        file.save(file_path)

        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)

        # Check if specified columns to anonymize are valid
        invalid_anonymize_columns = [
            col for col in columns_to_anonymize if col not in data.columns
        ]
        if invalid_anonymize_columns:
            return jsonify({
                "error": f"Invalid columns specified for anonymization: {', '.join(invalid_anonymize_columns)}"
            }), 400

        # Check if specified identifying attributes are valid
        invalid_identifying_columns = [
            col for col in identifying_attributes if col not in data.columns
        ]
        if invalid_identifying_columns:
            return jsonify({
                "error": f"Invalid identifying attributes specified: {', '.join(invalid_identifying_columns)}"
            }), 400

        # Filter the DataFrame to include only the columns to be synthesized
        data_to_anonymize = data[columns_to_anonymize]
        other_columns = data.drop(columns=columns_to_anonymize + identifying_attributes)

        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import TVAESynthesizer
        # Create SDV Metadata
        sdv_metadata = SingleTableMetadata()
        sdv_metadata.detect_from_dataframe(data_to_anonymize)

        print("Metadata detected from dataframe: %s", sdv_metadata.to_dict())

        # Initialize the TVAESynthesizer
        synthesizer = TVAESynthesizer(metadata=sdv_metadata)

        # Fit the synthesizer to the data to be anonymized
        synthesizer.fit(data_to_anonymize)

        # Generate synthetic data
        synthetic_data_to_anonymize = synthesizer.sample(num_rows=len(data_to_anonymize))

        # Combine synthetic data with unchanged columns
        synthetic_data = pd.concat(
            [other_columns.reset_index(drop=True),
             synthetic_data_to_anonymize.reset_index(drop=True)],
            axis=1,
        )

        # Save the synthetic data to a new CSV file
        synthetic_file_path = os.path.join(TEMP_DIR, f"synthetic_{file.filename}")
        synthetic_data.to_csv(synthetic_file_path, index=False)

        return send_file(synthetic_file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)
