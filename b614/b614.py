@app.route('/b614', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/b614/metadata', methods=['GET']) 
def get_metadata():
    return send_file('b614.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        