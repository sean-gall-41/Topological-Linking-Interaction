from generate_chain import generate_closed_chain, chain_to_JSON
from driver import N
from flask import Flask, jsonify, request, render_template
app = Flask(__name__)

@app.route('/hello', methods=['GET', 'POST'])

def data_helper():
    #POST request
    if request.method == 'POST':
        print('Incoming..')
        print(request.get_json()) #parse as JSON
        return 'OK', 200
    
    # GET request
    else:
        payload = chain_to_JSON(generate_closed_chain(N))
        return jsonify(payload)

@app.route('/test')
def test_page():
    # look inside 'templates' and serve 'index.html'
    return render_template('index.html')