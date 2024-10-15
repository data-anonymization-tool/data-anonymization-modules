@app.route('/anjana', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/anjana/metadata', methods=['GET']) 
def get_metadata():
    return send_file('anjana.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        