from flask import Flask, request, jsonify, make_response
from algorithm import edgeGetServer
from algorithm import configRWServer

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
    low_Threshold = (request.form.get('low_Threshold'))
    height_Threshold = (request.form.get('height_Threshold'))
    kernel_size = (request.form.get('kernel_size'))
    configRWServer.configfile_revise(
        [('edgeDetection', 'minThreshold', low_Threshold)])
    configRWServer.configfile_revise(
        [('edgeDetection', 'maxthreshold', height_Threshold)])
    configRWServer.configfile_revise(
        [('edgeDetection', 'kemelsize', kernel_size)])
    src = edgeGetServer.getSilhouette(
        file.read())
    res = make_response(jsonify({'code': 200, 'data': src}))
    return res


@app.route('/getFixContour', methods=['GET', 'POST'])
def getFixContour():
    file = request.files.get('file')
    points = request.form.get('points').split(',')
    pic_width = float(request.form.get('width'))
    pic_height = float(request.form.get('height'))
    leftTopP = [float(points[0]), float(points[1])]
    leftBottomP = [float(points[2]), float(points[3])]
    rightBottomP = [float(points[4]), float(points[5])]
    fy1, fy2, fy3, message, src = edgeGetServer.getFinalContour(
        file.read(), leftTopP, leftBottomP, rightBottomP, pic_width, pic_height)
    content = {
        "fy1": str(fy1),
        "fy2": str(fy2),
        "fy3": str(fy3),
        "message": message,
        "src": src
    }
    res = make_response(jsonify({'code': 200, 'data': content}))
    return res

# @app.route("/clustering", methods=['POST'])
# def clustering():
#     n_cluster = request.get_json()['n_cluster']
#     points = request.get_json()['points']
#     km = KMeans(n_clusters=n_cluster).fit(np.array(points))
#     labels = km.labels_
#     res = make_response(jsonify({'code': 200, 'data': labels.tolist()}))
#     return res


if __name__ == '__main__':
    app.run(host="0.0.0.0",
            port=5000)
