from flask import Flask, request, jsonify, make_response
from algorithm import edgeGetServer

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin'] = '*'
    environ.headers['Access-Control-Allow-Method'] = '*'
    environ.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return environ


@app.route("/upload_file", methods=['POST'])
def upload_file():
    file = request.files.get('file')
    res = make_response(jsonify({'code': 200, 'data': file.read().decode()}))
    return res


@app.route("/getSilhouette", methods=['POST'])
def getImageSilhouette():
    file = request.files.get('file')
    low_Threshold = int(request.form.get('low_Threshold'))
    height_Threshold = int(request.form.get('height_Threshold'))
    kernel_size = int(request.form.get('kernel_size'))
    src = edgeGetServer.getSilhouette(
        file.read(), low_Threshold, height_Threshold, kernel_size)
    res = make_response(jsonify({'code': 200, 'data': src}))
    return res


@app.route('/getFixContour', methods=['GET', 'POST'])
def getFixContour():
    file = request.files.get('file')
    fitting_strength = int(request.form.get('fitting_strength'))
    fy1, fy2, fy3, message, src, topLimit, leftContourLimit, rightContourLimit = edgeGetServer.getFinalContour(
        file.read(), fitting_strength)
    content = {
        "fy1": str(fy1),
        "fy2": str(fy2),
        "fy3": str(fy3),
        "limit": [str(topLimit), str(leftContourLimit), str(rightContourLimit)],
        "message": message,
        "src": src,
    }
    res = make_response(jsonify({'code': 200, 'data': content}))
    return res


@app.route('/drawRadiusPic', methods=['GET', 'POST'])
def drawRadiusPic():
    # print(request.get_json())
    fy1 = request.get_json()['fy1']
    fy2 = request.get_json()['fy2']
    midLineFactor = request.get_json()['fy3']
    limit = request.get_json()['limit']
    topLimit = int(limit[0])
    leftContourLimit = int(limit[1])
    rightContourLimit = int(limit[2])
    img_ori_width = float(request.get_json()['img_ori_width'])
    img_ori_height = float(request.get_json()['img_ori_height'])
    count = int(request.get_json()['count'])
    print(limit, topLimit, leftContourLimit, rightContourLimit)
    src, rList, yList = edgeGetServer.drawRadiusPic(count, img_ori_width, img_ori_height,
                                                    fy1, fy2, midLineFactor,
                                                    topLimit, leftContourLimit, rightContourLimit)
    content = {
        "src": src,
        'r': rList,
        'y': yList,
        "message": "ok",
        'code': 200
    }
    res = make_response(jsonify({'code': 200, 'data': content}))
    return res


@app.route('/runAllProcess', methods=['POST'])
def runAllProcess():
    file = request.files.get('file')
    low_Threshold = int(request.form.get('low_Threshold'))
    height_Threshold = int(request.form.get('height_Threshold'))
    kernel_size = int(request.form.get('kernel_size'))
    fitting_strength = int(request.form.get('fitting_strength'))
    count = int(request.form.get('count'))
    src1, src2, src3, fy1, fy2, fy3, topLimit, leftContourLimit, rightContourLimit, width, height, rList, yList \
        = edgeGetServer.runAll(file.read(), low_Threshold, height_Threshold, fitting_strength, count, kernel_size)
    content = {
        "src1": src1,
        "src2": src2,
        "src3": src3,
        "fy1": str(fy1),
        "fy2": str(fy2),
        "fy3": str(fy3),
        "topLimit": int(topLimit),
        "leftLimit": int(leftContourLimit),
        "rightLimit": int(rightContourLimit),
        "width": int(width),
        "height": int(height),
        'r': rList,
        'y': yList,
        "message": "ok",
        'code': 200
    }
    res = make_response(jsonify({'code': 200, 'data': content}))
    return res

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

