@app.route('/aggregationbla', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/aggregationbla/metadata', methods=['GET']) 
def get_metadata():
    return send_file('aggregationbla.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        