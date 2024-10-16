@app.route('/youhello', methods=['POST'])

def anonymize():
# Add your anonymization logic here

@app.route('/youhello/metadata', methods=['GET']) 
def get_metadata():
    return send_file('youhello.json', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
        