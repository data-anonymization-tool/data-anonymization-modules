@app.route('/loader2', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/loader2/metadata', methods=['GET']) 
def get_metadata():
    return send_file('loader2.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        