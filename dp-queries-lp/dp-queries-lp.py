import pandas as pd
from diffprivlib.mechanisms import Laplace
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_file(file):
    try:
        filename = file.filename
        
        # Determine file type from extension
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

def count_query(df, condition, epsilon):
    original_count = sum(condition(row) for _, row in df.iterrows())
    laplace_mech = Laplace(epsilon=epsilon, sensitivity=1)
    noisy_count = laplace_mech.randomise(original_count)
    return original_count, noisy_count

def sum_query(df, column, epsilon):
    total_sum = df[column].sum()
    sensitivity = df[column].max()  # Sensitivity for sum queries
    laplace_mech = Laplace(epsilon=epsilon, sensitivity=sensitivity)
    noisy_sum = laplace_mech.randomise(total_sum)
    return total_sum, noisy_sum

def mean_query(df, column, epsilon):
    count = df[column].shape[0]
    total_sum = df[column].sum()
    sensitivity_sum = df[column].max()
    sensitivity_count = 1  # Sensitivity for count queries

    laplace_sum = Laplace(epsilon=epsilon/2, sensitivity=sensitivity_sum)
    laplace_count = Laplace(epsilon=epsilon/2, sensitivity=sensitivity_count)

    noisy_sum = laplace_sum.randomise(total_sum)
    noisy_count = laplace_count.randomise(count)

    noisy_mean = noisy_sum / noisy_count
    return total_sum / count, noisy_mean

def median_query(df, column, epsilon):
    sorted_data = sorted(df[column])
    num_data = len(sorted_data)
    median = sorted_data[num_data // 2] if num_data % 2 != 0 else (sorted_data[num_data // 2 - 1] + sorted_data[num_data // 2]) / 2
    laplace_mech = Laplace(epsilon=epsilon, sensitivity=num_data)
    noisy_median = laplace_mech.randomise(median)  
    return median, noisy_median

def mode_query(df, column, epsilon):
    mode_value = df[column].mode().values[0]
    count_mode = df[column].value_counts().max()
    laplace_mech = Laplace(epsilon=epsilon, sensitivity=count_mode)
    noisy_mode = laplace_mech.randomise(mode_value)
    return mode_value, noisy_mode

def variance_query(df, column, epsilon):
    mean = df[column].mean()
    squared_deviations = ((df[column] - mean) ** 2).sum()
    count = df[column].shape[0]
    variance = squared_deviations / count
    sensitivity = max(df[column].max() - df[column].min(), 1)  # Sensitivity for variance

    laplace_mech = Laplace(epsilon=epsilon, sensitivity=sensitivity)
    noisy_variance = laplace_mech.randomise(variance)

    return variance, noisy_variance

def std_dev_query(df, column, epsilon):
    _, noisy_variance = variance_query(df, column, epsilon)
    noisy_std_dev = noisy_variance ** 0.5
    return noisy_std_dev

@app.route('/dp-queries-lp/metadata', methods=['GET'])
def get_metadata():
    return send_file('dp-queries-lp.json', as_attachment=False)

@app.route('/dp-queries-lp/', methods=['POST'])
def all():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    condition_value = float(request.form.get('Condition Value'))
    epsilon = float(request.form.get('Epsilon'))

    try:
        df = load_file(file)
        condition = lambda row: row[column] > condition_value
        original_count, noisy_count = count_query(df, condition, epsilon)
        original_sum, noisy_sum = sum_query(df, column, epsilon)
        original_mean, noisy_mean = mean_query(df, column, epsilon)
        original_median, noisy_median = median_query(df, column, epsilon)
        original_mode, noisy_mode = mode_query(df, column, epsilon)
        original_variance, noisy_variance = variance_query(df, column, epsilon)
        noisy_std_dev = std_dev_query(df, column, epsilon)
        return jsonify({
            "original_count": int(original_count),
            "noisy_count": int(noisy_count),
            "original_sum": float(original_sum),
            "noisy_sum": float(noisy_sum),
            "original_mean": float(original_mean),
            "noisy_mean": float(noisy_mean),
            "original_median": float(original_median),
            "noisy_median": float(noisy_median),
            "original_mode": float(original_mode),
            "noisy_mode": float(noisy_mode),
            "original_variance": float(original_variance),
            "noisy_variance": float(noisy_variance),
            "noisy_std_dev": float(noisy_std_dev)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dp-queries-lp/count/', methods=['POST'])
def count():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    condition_value = float(request.form.get('Condition Value'))
    epsilon = float(request.form.get('Epsilon'))

    try:
        df = load_file(file)
        condition = lambda row: row[column] > condition_value
        original_count, noisy_count = count_query(df, condition, epsilon)
        return jsonify({
            "original_count": int(original_count),
            "noisy_count": int(noisy_count)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dp-queries-lp/sum/', methods=['POST'])
def sum_():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon'))

    try:
        df = load_file(file)
        original_sum, noisy_sum = sum_query(df, column, epsilon)
        return jsonify({
            "original_sum": float(original_sum),
            "noisy_sum": float(noisy_sum)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dp-queries-lp/mean/', methods=['POST'])
def mean():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon'))

    try:
        df = load_file(file)
        original_mean, noisy_mean = mean_query(df, column, epsilon)
        return jsonify({
            "original_mean": float(original_mean),
            "noisy_mean": float(noisy_mean)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dp-queries-lp/median/', methods=['POST'])
def median():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon'))

    try:
        df = load_file(file)
        original_median, noisy_median = median_query(df, column, epsilon)
        return jsonify({
            "original_median": float(original_median),
            "noisy_median": float(noisy_median)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dp-queries-lp/mode/', methods=['POST'])
def mode():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon'))

    try:
        df = load_file(file)
        original_mode, noisy_mode = mode_query(df, column, epsilon)
        return jsonify({
            "original_mode": float(original_mode),
            "noisy_mode": float(noisy_mode)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dp-queries-lp/variance/', methods=['POST'])
def variance():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon'))

    try:
        df = load_file(file)
        original_variance, noisy_variance = variance_query(df, column, epsilon)
        return jsonify({
            "original_variance": float(original_variance),
            "noisy_variance": float(noisy_variance)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dp-queries-lp/std_dev/', methods=['POST'])
def std_dev():
    file = request.files['file']
    column = request.form.get('Column to be anonymized')
    epsilon = float(request.form.get('Epsilon'))

    try:
        df = load_file(file)
        noisy_std_dev = std_dev_query(df, column, epsilon)
        return jsonify({
            "noisy_std_dev": noisy_std_dev
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port = 5010, debug=True)