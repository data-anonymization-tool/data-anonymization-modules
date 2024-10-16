@app.route('/bhello5', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/bhello5/metadata', methods=['GET']) 
def get_metadata():
    return send_file('bhello5.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        