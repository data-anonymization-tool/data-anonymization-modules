@app.route('/nisha', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/nisha/metadata', methods=['GET']) 
def get_metadata():
    return send_file('nisha.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        