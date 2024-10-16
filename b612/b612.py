@app.route('/b612', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/b612/metadata', methods=['GET']) 
def get_metadata():
    return send_file('b612.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        