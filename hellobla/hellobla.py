@app.route('/hellobla', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/hellobla/metadata', methods=['GET']) 
def get_metadata():
    return send_file('hellobla.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        