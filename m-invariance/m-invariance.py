@app.route('/m-invariance', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/m-invariance/metadata', methods=['GET']) 
def get_metadata():
    return send_file('m-invariance.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        