from flask import Flask, request, jsonify, send_file
import pandas as pd
from werkzeug.utils import secure_filename
import os
import uuid
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define upload and anonymized file directories
UPLOAD_FOLDER = './uploads'
ANONYMIZED_FOLDER = './anonymized'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANONYMIZED_FOLDER'] = ANONYMIZED_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANONYMIZED_FOLDER, exist_ok=True)

# Function to detect categorical columns
def detect_categorical_columns(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return set(categorical_columns)

# Function to get unique value count for a column
def diversity(df, partition, column):
    return len(df[column][partition].unique())

def is_l_diverse(df, partition, sensitive_column, l):
    return diversity(df, partition, sensitive_column) >= l

# Function to get spans for each column
def get_spans(df, partition, scale=None):
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max() - df[column][partition].min()
        if scale is not None:
            span /= scale[column]
        spans[column] = span
    return spans

# Function to split a partition into two based on a column
def split(df, partition, column):
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values) // 2])
        rv = set(values[len(values) // 2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return dfl, dfr

# Function to partition the dataset into k-anonymous and l-diverse subsets
def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid, k, l):
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column, k, l) or not is_valid(df, rp, sensitive_column, k, l):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions

# Function to check if a partition is k-anonymous
def is_k_anonymous(df, partition, sensitive_column, k):
    return len(partition) >= k

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
    aggregations = {col: agg_categorical_column if col in categorical else agg_numerical_column for col in feature_columns}
    rows = []
    for i, partition in enumerate(partitions):
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

def remove_identifier_columns(df, identifier_columns):
    """Remove identifier columns from the DataFrame."""
    return df.drop(columns=identifier_columns, errors='ignore')

@app.route('/l-diversity/metadata', methods=['GET'])
def get_metadata():
    return send_file('l-diversity.json', as_attachment=False)

@app.route('/l-diversity/', methods=['POST'])
def anonymize():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Get parameters from the request
    k = int(request.form.get('k', 3))
    l = int(request.form.get('l', 2))
    identifier_columns = [col.strip() for col in request.form.get('Direct Identifier Columns', '').split(',')]
    feature_columns = [col.strip() for col in request.form.get('Quasi Identifier Columns', '').split(',')]
    sensitive_column = request.form.get('Column to be anonymized').strip()

    # Load and preprocess the dataset
    df = pd.read_csv(file_path)
    df = remove_identifier_columns(df, identifier_columns)

    global categorical
    categorical = detect_categorical_columns(df)

    for name in categorical:
        df[name] = df[name].astype('category')

    # Get spans for the dataset
    full_spans = get_spans(df, df.index)

    # Apply partitioning method
    finished_l_diverse_partitions = partition_dataset(
        df, feature_columns, sensitive_column, full_spans,
        lambda *args: is_k_anonymous(*args) and is_l_diverse(*args),
        k, l
    )

    # Build the anonymized dataset
    dfl = build_anonymized_dataset(df, finished_l_diverse_partitions, feature_columns, sensitive_column)

    # Save the anonymized dataset to a file
    anonymized_filename = f"anonymized_{uuid.uuid4().hex}_{filename}"
    anonymized_filepath = os.path.join(app.config['ANONYMIZED_FOLDER'], anonymized_filename)
    dfl.to_csv(anonymized_filepath, index=False)

    return send_file(anonymized_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)