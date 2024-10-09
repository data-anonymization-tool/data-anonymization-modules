from flask import Flask, request, jsonify, send_file
import pandas as pd
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__)

# Define upload and anonymized file directories
UPLOAD_FOLDER = './uploads'
ANONYMIZED_FOLDER = './anonymized'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANONYMIZED_FOLDER'] = ANONYMIZED_FOLDER

# Create directories if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(ANONYMIZED_FOLDER):
    os.makedirs(ANONYMIZED_FOLDER)

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
    return ','.join(set(series))

# Function to aggregate numerical columns
def agg_numerical_column(series):
    return series.mean()

# Function to convert numpy types to native Python types
def convert_to_native_type(value):
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
            print(f"Finished {i} partitions...")
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].agg(aggregations)
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

@app.route('/k-anonymity', methods=['POST'])
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
        file.save(file_path)

        # Get the parameters from the request
        k = int(request.form.get('k', 3))
        feature_columns = request.form.getlist('feature_columns')
        sensitive_column = request.form.get('sensitive_column')

        # Load the dataset
        df = pd.read_csv(file_path)

        global categorical
        categorical = detect_categorical_columns(df)

        for name in categorical:
            df[name] = df[name].astype('category')

        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Convert categorical columns to category type
        for column in categorical_columns:
            df[column] = df[column].astype('category')
        
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
        
        # Adding a comment to check update

if __name__ == '__main__':
    app.run(debug=True)
