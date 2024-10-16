@app.route('/b2', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/b2/metadata', methods=['GET']) 
def get_metadata():
    return send_file('b2.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        