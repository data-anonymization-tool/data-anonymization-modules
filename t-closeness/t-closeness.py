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
            span = len(df.loc[partition, column].unique())
        else:
            span = df.loc[partition, column].max() - df.loc[partition, column].min()
        if scale is not None:
            span = span / scale[column]
        spans[column] = span
    return spans

# Function to split a partition into two based on a column
def split(df, partition, column):
    dfp = df.loc[partition, column]
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

# Function to partition the dataset into k-anonymous subsets
def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid, global_freqs,k, t):
    finished_partitions = []
    partitions = [df.index]
    print("In partition_dataset: ",global_freqs)
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column, global_freqs,k, t) or not is_valid(df, rp, sensitive_column, global_freqs, k, t):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions

# Function to check if a partition is k-anonymous
def is_k_anonymous(df, partition, sensitive_column, global_freqs, k, t):
    if len(partition) < k:
        return False
    return True

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

def t_closeness(df, partition, column, global_freqs):
    total_count = float(len(partition))
    d_max = None
    group_counts = df.loc[partition].groupby(column, observed=False)[column].agg('count')
    for value, count in group_counts.to_dict().items():
        t = count / total_count
        d = abs(t - global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max

def is_t_close(df, partition, sensitive_column, global_freqs, k, t):
    print("In is_t_close: ",global_freqs)
    if not sensitive_column in categorical:
        raise ValueError("This method only works for categorical values.")
    return t_closeness(df, partition, sensitive_column, global_freqs) <= t

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

def remove_identifier_columns(df, identifier_columns):
    """Remove identifier columns from the DataFrame."""
    return df.drop(columns=identifier_columns, errors='ignore')

@app.route('/t-closeness/metadata', methods=['GET'])
def get_metadata():
    return send_file('t-closeness.json', as_attachment=False)

@app.route('/t-closeness/', methods=['POST'])
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
        t = float(request.form.get('t', 0.2))
        identifier_columns = [col.strip() for col in request.form.get('Direct Identifier Columns', '').split(',')]
        feature_columns = [col.strip() for col in request.form.get('Quasi Identifier Columns', '').split(',')]
        sensitive_column = request.form.get('Column to be anonymized').strip()

        # Load the dataset
        df = pd.read_csv(file_path)

        df = remove_identifier_columns(df, identifier_columns)

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

        # Generate the global frequencies for the sensitive column 
        global_freqs = {}
        total_count = float(len(df))
        group_counts = df.groupby(sensitive_column, observed=False)[sensitive_column].agg('count')
        
        for value, count in group_counts.to_dict().items():
            t = count / total_count
            global_freqs[value] = t

        # Partition dataset into k-anonymous subsets with t-closeness
        finished_t_close_partitions = partition_dataset(df, feature_columns, 
                                                        sensitive_column, 
                                                        full_spans, 
                                                        lambda *args: is_k_anonymous(*args) and 
                                                                      is_t_close(*args),global_freqs,k,t)

        # Build anonymized dataset from the k-anonymous partitions
        dft = build_anonymized_dataset(df, finished_t_close_partitions, feature_columns, sensitive_column)

        # Save the anonymized dataset to a file
        anonymized_filename = f"anonymized_{uuid.uuid4().hex}_{filename}"
        anonymized_filepath = os.path.join(app.config['ANONYMIZED_FOLDER'], anonymized_filename)
        dft.to_csv(anonymized_filepath, index=False)

        # Send the anonymized file as a response
        return send_file(anonymized_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
