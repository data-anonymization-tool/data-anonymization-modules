@app.route('/beyonce', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/beyonce/metadata', methods=['GET']) 
def get_metadata():
    return send_file('beyonce.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        