@app.route('/b3', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/b3/metadata', methods=['GET']) 
def get_metadata():
    return send_file('b3.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        