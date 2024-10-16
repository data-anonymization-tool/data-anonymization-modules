@app.route('/loader', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/loader/metadata', methods=['GET']) 
def get_metadata():
    return send_file('loader.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        