from flask import Flask, request, jsonify

app = Flask(__name__)

def run_my_script(input_data):
    # reverses a string
    output_data = input_data[::-1]
    return output_data

@app.route('/myendpoint', methods=['POST'])
def myendpoint():
    input_data = request.json
    output_data = run_my_script(input_data)
    return jsonify({'result' : output_data})

if __name__ == '__main__':
    app.run()