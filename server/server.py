from flask import Flask, request, jsonify, make_response
import time
from algorithm import edgeGetServer

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin'] = '*'
    environ.headers['Access-Control-Allow-Method'] = '*'
    environ.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return environ


@app.route('/runAllProcess', methods=['POST'])
def runAllProcess():
    startTime=time.time()
    file = request.files.get('file')
    low_Threshold = int(request.form.get('low_Threshold'))
    height_Threshold = int(request.form.get('height_Threshold'))
    kernel_size = int(request.form.get('kernel_size'))
    fitting_strength = int(request.form.get('fitting_strength'))
    count = int(request.form.get('count'))
    src1, src2, src3, dList, arcLengthList, nozzleDiameter \
        = edgeGetServer.runAll(file.read(), low_Threshold, height_Threshold, fitting_strength, count, kernel_size)
    content = {
        "src1": src1,
        "src2": src2,
        "src3": src3,
        "nozzleDiameter": int(nozzleDiameter),
        'r': dList,
        'y': arcLengthList,
        "message": "ok",
        'code': 200
    }
    res = make_response(jsonify({'code': 200, 'data': content}))
    endTime=time.time()
    # print((endTime - startTime)*1000)
    return res


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
