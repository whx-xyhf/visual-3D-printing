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
    src = edgeGetServer.getSilhouette(file.read(), low_Threshold, height_Threshold, kernel_size)
    res = make_response(jsonify({'code': 200, 'data': src}))
    return res

@app.route('/getFixContour', methods=['GET', 'POST'])
def getFixContour():
    file = request.files.get('file')
    points = request.form.get('points').split(',')
    pic_width = float(request.form.get('width'))
    pic_height = float(request.form.get('height'))
    # leftTopP = [float(points[0]), float(points[1])]
    # leftBottomP = [float(points[2]), float(points[3])]
    # rightBottomP = [float(points[4]), float(points[5])]
    fitting_strength = int(request.form.get('fitting_strength'))
    fy1, fy2, fy3, message, src = edgeGetServer.getFinalContour(file.read(), fitting_strength)
    content = {
        "fy1": str(fy1),
        "fy2": str(fy2),
        "fy3": str(fy3),
        "message": message,
        "src": src,
    }
    res = make_response(jsonify({'code': 200, 'data': content}))
    return res

# @app.route('/dataExport', methods=['GET', 'POST'])
# def dataExport():
#     fy1 = (request.form['fy1'])
#     fy2 = (request.form['fy2'])
#     midLineFactor = (request.form['fy3'])
#     content = {
#         "rList": edgeGetServer.dataExport(fy1, fy2, midLineFactor),
#         "message": "ok",
#     }
#     res = make_response(jsonify({'code': 200, 'data': content}))
#     return res


@app.route('/drawRadiusPic', methods=['GET', 'POST'])
def drawRadiusPic():
    # print(request.get_json())
    fy1 = request.get_json()['fy1']
    fy2 = request.get_json()['fy2']
    midLineFactor = request.get_json()['fy3']
    points = request.get_json()['points']
    leftTopP = points[0]
    leftBottomP = points[1]
    rightBottomP = points[2]
    img_ori_width = float(request.get_json()['img_ori_width'])
    img_ori_height = float(request.get_json()['img_ori_height'])
    pic_show_width = float(request.get_json()['show_width'])
    pic_show_height = float(request.get_json()['show_height'])
    count = int(request.get_json()['count'])
    src, rList, yList = edgeGetServer.drawRadiusPic(count, img_ori_width, img_ori_height,
                                                    pic_show_width, pic_show_height,
                                                    fy1, fy2, midLineFactor,
                                                    leftTopP, leftBottomP, rightBottomP)
    content = {
        "src": src,
        'r': rList,
        'y': yList,
        "message": "ok",
        'code': 200
    }
    res = make_response(jsonify({'code': 200, 'data': content}))
    return res


if __name__ == '__main__':
    app.run(host="0.0.0.0",
            port=5000)
