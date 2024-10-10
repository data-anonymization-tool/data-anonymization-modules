import pandas as pd
from diffprivlib.mechanisms import Exponential, Laplace
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_file(file):
    try:
        filename = file.filename
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
        
        return df
    
    except UnicodeDecodeError:
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file, encoding='ISO-8859-1')
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file)
            return df
        except Exception as e:
            raise Exception(f"An error occurred while reading the file: {e}")
    
    except pd.errors.ParserError as e:
        raise Exception(f"Error parsing file: {e}")
    
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def ensure_categorical(df, column):
    if not isinstance(df[column].dtype, pd.CategoricalDtype):
        df[column] = df[column].astype('category')
    return df

def utility_function(category, df, column):
    return df[df[column] == category].shape[0]

def differential_private_frequency(df, column, epsilon):
    categories = df[column].unique()
    laplace_mechanism = Laplace(epsilon=epsilon, sensitivity=1)
    frequency_estimates = {}
    
    for category in categories:
        count = df[df[column] == category].shape[0]
        noisy_count = max(0, count + laplace_mechanism.randomise(0))
        frequency_estimates[category] = int(noisy_count)
    
    return frequency_estimates

def differential_private_mode_and_majority_vote(df, column, epsilon):
    categories = df[column].unique()
    
    # Calculate the utilities (i.e., frequencies) for each category
    utilities = [utility_function(category, df, column) for category in categories]
    
    # Use the Exponential mechanism with sensitivity=1 and the utility list
    exp_mechanism = Exponential(epsilon=epsilon, sensitivity=1, utility=utilities)
    
    # Randomize to select the category with the highest utility
    selected_index = exp_mechanism.randomise()
    
    # Select the category with the highest utility
    mode = categories[selected_index]
    majority_vote = mode  # Majority vote is the same as the mode in this context

    return mode, majority_vote

def differential_private_top_k(df, column, epsilon, k):
    categories = df[column].unique()
    
    # Calculate the utilities (frequencies) for each category
    utilities = [utility_function(category, df, column) for category in categories]
    
    # Sort utilities in descending order and pick top k
    top_k_indices = np.argsort(utilities)[-k:][::-1]
    top_k = {categories[i]: utilities[i] for i in top_k_indices}

    return top_k

def differential_private_entropy(df, column, epsilon):
    frequency_estimates = differential_private_frequency(df, column, epsilon)
    total_count = sum(frequency_estimates.values())
    entropy = 0.0
    
    for count in frequency_estimates.values():
        p = count / total_count if total_count > 0 else 0
        if p > 0:
            entropy -= p * np.log(p)
    
    return entropy

def differential_private_contingency_table(df, column1, column2, epsilon):
    if column2 not in df.columns:
        return jsonify({"error": f"Column '{column2}' not found in the file."}), 400

    df = ensure_categorical(df, column2)

    table = pd.crosstab(df[column1], df[column2])
    noisy_table = table.copy()
    laplace_mechanism = Laplace(epsilon=epsilon, sensitivity=1)
    
    for i in table.index:
        for j in table.columns:
            count = table.loc[i, j]
            noisy_count = max(0, count + laplace_mechanism.randomise(0))
            noisy_table.loc[i, j] = int(noisy_count)
    
    # Convert the DataFrame to a dictionary for JSON serialization
    return noisy_table.to_dict()

@app.route('/dp-queries-ep/metadata', methods=['GET'])
def get_metadata():
    return send_file('dp-queries-ep.json', as_attachment=False)

@app.route('/dp-queries-ep/frequency/', methods=['POST'])
def frequency():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon', 1.0))

    try:
        df = load_file(file)
        
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found in the file."}), 400
        if pd.api.types.is_numeric_dtype(df[column]):
            return jsonify({"error": f"Column '{column}' is numerical. Please use Laplace mechanism for numerical data."}), 400
        
        df = ensure_categorical(df, column)
        dp_frequency = differential_private_frequency(df, column, epsilon)
        
        response_data = {
            "Differentially Private Frequency Estimation": dp_frequency
        }  
        return jsonify(response_data), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dp-queries-ep/mode/', methods=['POST'])
def mode():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon', 1.0))

    try:
        df = load_file(file)
        
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found in the file."}), 400
        if pd.api.types.is_numeric_dtype(df[column]):
            return jsonify({"error": f"Column '{column}' is numerical. Please use Laplace mechanism for numerical data."}), 400
        
        df = ensure_categorical(df, column)
        mode, majority_vote = differential_private_mode_and_majority_vote(df, column, epsilon)
        
        response_data = {
            "Mode": mode,
            "Majority Vote": majority_vote
        }  
        return jsonify(response_data), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/dp-queries-ep/entropy/', methods=['POST'])
def entropy():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon', 1.0))

    try:
        df = load_file(file)
        
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found in the file."}), 400
        if pd.api.types.is_numeric_dtype(df[column]):
            return jsonify({"error": f"Column '{column}' is numerical. Please use Laplace mechanism for numerical data."}), 400
        
        df = ensure_categorical(df, column)
        dp_entropy = differential_private_entropy(df, column, epsilon)
        
        response_data = {
            "Differentially Private Entropy Calculation": dp_entropy
        }  
        return jsonify(response_data), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/dp-queries-ep/top-k/', methods=['POST'])
def topk():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon', 1.0))
    k = int(request.form.get('k', 0))

    try:
        df = load_file(file)
        
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found in the file."}), 400
        if pd.api.types.is_numeric_dtype(df[column]):
            return jsonify({"error": f"Column '{column}' is numerical. Please use Laplace mechanism for numerical data."}), 400
        
        df = ensure_categorical(df, column)
        topkselection = differential_private_top_k(df, column, epsilon, k) 
        if k > 0:
            response_data= {
                "Differentially Private Top-k Selection": topkselection
            }
            return jsonify(response_data), 200
        else:
            return jsonify({"error":f"k value missing or must be positive"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dp-queries-ep/contingency/', methods=['POST'])
def contingency():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon', 1.0))
    column2 = request.form.get('Column 2')

    try:
        df = load_file(file)
        
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found in the file."}), 400
        if pd.api.types.is_numeric_dtype(df[column]):
            return jsonify({"error": f"Column '{column}' is numerical. Please use Laplace mechanism for numerical data."}), 400
        
        df = ensure_categorical(df, column)
        contingency = differential_private_contingency_table(df, column, column2, epsilon)   
        if column2:
            response_data= {
                "Differentially Private Contingency Table": contingency
            }
            return jsonify(response_data), 200
        else:
            return jsonify({"error":f"Second column missing"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dp-queries-ep/', methods=['POST'])
def all():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon', 1.0))
    k = int(request.form.get('k', 0))
    column2 = request.form.get('Column 2')

    try:
        df = load_file(file)
        
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found in the file."}), 400
        if pd.api.types.is_numeric_dtype(df[column]):
            return jsonify({"error": f"Column '{column}' is numerical. Please use Laplace mechanism for numerical data."}), 400
        
        df = ensure_categorical(df, column)
        dp_frequency = differential_private_frequency(df, column, epsilon)
        mode, majority_vote = differential_private_mode_and_majority_vote(df, column, epsilon)
        dp_entropy = differential_private_entropy(df, column, epsilon)
        
        response_data = {
            "Differentially Private Frequency Estimation": dp_frequency,
            "Mode": mode,
            "Majority Vote": majority_vote,
            "Differentially Private Entropy Calculation": dp_entropy
        }
        if k > 0:
            response_data["Differentially Private Top-k Selection"] = differential_private_top_k(df, column, epsilon, k)
        if column2:
            response_data["Differentially Private Contingency Table"] = differential_private_contingency_table(df, column, column2, epsilon)       
        return jsonify(response_data), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)