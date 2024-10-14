from flask import Flask, request, jsonify, send_file
import pandas as pd
from werkzeug.utils import secure_filename
import os
import uuid
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define upload and anonymized file directories
UPLOAD_FOLDER = './uploads'
ANONYMIZED_FOLDER = './anonymized'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANONYMIZED_FOLDER'] = ANONYMIZED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file size limit

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANONYMIZED_FOLDER, exist_ok=True)

# Function to detect categorical columns
def detect_categorical_columns(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return set(categorical_columns)

# Function to get spans for each column
def get_spans(df, partition, scale=None):
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max() - df[column][partition].min()
        if scale is not None:
            span = span / scale[column]
        spans[column] = span
    return spans

# Function to split a partition into two based on a column
def split(df, partition, column):
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:        
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)

# Function to check if a partition is k-anonymous
def is_k_anonymous(df, partition, sensitive_column, k):
    if len(partition) < k:
        return False
    return True

# Function to partition the dataset into k-anonymous subsets
def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid, k):
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column, k) or not is_valid(df, rp, sensitive_column, k):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions

# Function to aggregate categorical columns
def agg_categorical_column(series):
    # Instead of concatenating all unique values, use the most frequent (mode)
    return series.mode().iloc[0] if not series.empty else None

# Function to aggregate numerical columns
def agg_numerical_column(series):
    return series.mean()

# Function to convert numpy types to native Python types
def convert_to_native_type(value):
    if isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
        # Handle cases where we have more than one value
        if value.size == 1:
            return value.item()  # Safe to convert to scalar
        else:
            return list(value)  # Return as a list for multi-value cases
    if hasattr(value, "item"):
        return value.item()
    return value

# Function to build the anonymized dataset
def build_anonymized_dataset(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    rows = []
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            logging.info(f"Finished {i} partitions...")
        if max_partitions is not None and i > max_partitions:
            break
        # Perform aggregation for the partition
        try:
            grouped_columns = df.loc[partition].agg(aggregations)
        except ValueError:
            # Handle categorical columns separately with transform if necessary
            grouped_columns = df.loc[partition].transform(aggregations)
        grouped_columns = {col: convert_to_native_type(value) for col, value in grouped_columns.items()}
        sensitive_counts = df.loc[partition].groupby(sensitive_column).size()
        for sensitive_value, count in sensitive_counts.items():
            if count == 0:
                continue
            row = grouped_columns.copy()
            row.update({
                sensitive_column: sensitive_value,
                'count': count,
            })
            rows.append(row)
    return pd.DataFrame(rows)

def remove_identifier_columns(df, identifier_columns):
    """
    Remove identifier columns from the DataFrame.

    :param df: Pandas DataFrame from which to remove columns
    :param identifier_columns: List of column names to remove from the DataFrame
    :return: DataFrame with identifier columns removed
    """
    # Remove the specified identifier columns from the DataFrame
    df = df.drop(columns=identifier_columns, errors='ignore')
    
    return df

@app.route('/k-anonymity/metadata', methods=['GET'])
def get_metadata():
    return send_file('k-anonymity.json', as_attachment=False)

@app.route('/k-anonymity/', methods=['POST'])
def anonymize():
    # Check if the file part is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    # Check if a file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Ensure path is within allowed directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(app.config['UPLOAD_FOLDER'])):
            return jsonify({"error": "Invalid file path"}), 400
        
        file.save(file_path)

        # Get the parameters from the request
        try:
            k = int(request.form.get('k', 3))
            if k <= 0:
                raise ValueError
        except ValueError:
            return jsonify({"error": "Invalid value for k. Must be a positive integer."}), 400

        identifier_columns = [col.strip() for col in request.form.getlist('Direct Identifier Columns', '').split(',')]
        feature_columns = [col.strip() for col in request.form.get('Quasi Identifier Columns', '').split(',')]
        sensitive_column = request.form.get('Column to be anonymized').strip()

        # Load the dataset
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Loaded DataFrame columns: {df.columns.tolist()}")
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            return jsonify({"error": "Failed to read the dataset. Please check the file format."}), 400
        
        # Remove the identifier columns
        df = remove_identifier_columns(df, identifier_columns)
        logging.info(f"DataFrame columns after removing identifiers: {df.columns.tolist()}")

        # Check if sensitive column and feature columns exist
        missing_columns = []
        if sensitive_column not in df.columns:
            missing_columns.append(sensitive_column)
        for col in feature_columns:
            if col not in df.columns:
                missing_columns.append(col)

        if missing_columns:
            logging.error(f"Missing columns: {missing_columns}. Available columns: {df.columns.tolist()}")
            return jsonify({"error": "Sensitive or feature columns not found in dataset."}), 400

        global categorical
        categorical = detect_categorical_columns(df)

        for name in categorical:
            df[name] = df[name].astype('category')

        # Get the full spans for the dataset
        full_spans = get_spans(df, df.index)

        # Partition the dataset
        finished_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, is_k_anonymous, k)

        # Build the anonymized dataset
        dfn = build_anonymized_dataset(df, finished_partitions, feature_columns, sensitive_column)

        # Save the anonymized dataset to a file
        anonymized_filename = f"anonymized_{uuid.uuid4().hex}_{filename}"
        anonymized_filepath = os.path.join(app.config['ANONYMIZED_FOLDER'], anonymized_filename)
        dfn.to_csv(anonymized_filepath, index=False)

        # Send the anonymized file as a response
        return send_file(anonymized_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
