@app.route('/yh22', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/yh22/metadata', methods=['GET']) 
def get_metadata():
    return send_file('yh22.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        