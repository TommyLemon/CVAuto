import threading

import requests
from flask import Flask, request, jsonify, Response, make_response
from flask_cors import CORS, cross_origin
from pandas.io.common import is_url

from ultralytics import YOLO
from ultralytics.utils.plotting import Colors

import os
import base64
from io import BytesIO

from unitauto.methodutil import not_empty, not_none, is_empty, null, is_none, size, false, true, KEY_CODE, KEY_MSG, \
    KEY_OK, CODE_SUCCESS, MSG_SUCCESS, KEY_THROW, KEY_TRACE
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# 设置文件上传的存储路径和允许的文件类型
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

lock = threading.Lock()
# YOLO 模型加载
model = YOLO('yolo11m.pt')  # 使用你选择的模型
pose_model = YOLO('yolo11m-pose.pt')  # 使用你选择的模型
seg_model = YOLO('yolo11m-seg.pt')  # 使用你选择的模型
names = model.names  # 获取类别名称映射
colors = Colors()

# 检查文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 将 Base64 字符串转换为图像
def decode_base64_image(base64_string):
    try:
        # 处理 Base64 编码的图像
        img_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(img_data))
        return np.array(image)
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


# 下载 URL 图像并转换为模型支持的格式
def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果响应失败，抛出异常
        img = Image.open(BytesIO(response.content))  # 从响应内容创建图片
        return np.array(img)  # 转换为 NumPy 数组
    except Exception as e:
        print(f"Error downloading or processing image: {e}")
        return None

# 处理推理请求
def cors_response(data):
    host = request.headers.get('Origin') or request.headers.get('Referer') or 'http://localhost'

    rsp = make_response(jsonify(data), 200)
    # rsp.status = status
    # rsp.status_code = status
    rsp.headers.add('Access-Control-Allow-Origin', host)  # 允许所有域名访问
    rsp.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')  # 允许的请求方法
    rsp.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Authentication,Cookie,Token,JWT')  # 允许的请求头
    rsp.headers.add('Access-Control-Allow-Credentials', 'true')  # 支持 cookies 或者认证信息
    rsp.headers.add('Access-Control-Max-Age', '3600')  # 预检请求结果缓存时间，单位为秒
    return rsp


@app.route('/predict', methods=['POST', 'OPTIONS'])
# @cross_origin
def predict():
    if request.method == 'OPTIONS':
        return cors_response({})

    imgs = []

    files = request.files
    if 'file' in files:  # 处理文件上传
        # 检查文件是否在请求中
        file = files['file']
        if file.filename == '':
            return cors_response({
            KEY_OK: false,
            KEY_CODE: 400,
            KEY_MSG: 'No selected file'
        })

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # 使用 OpenCV 读取文件并进行推理
            img = cv2.imread(file_path)
            if not_none(img):
                imgs.append(img)

    try:
        data = request.get_json()  # 获取 JSON 请求数据
        # 处理 Base64 编码的图像
        s = data.get('image')
        if is_url(s):
            img = download_image(s)
        else:
            img = null if is_empty(s) else decode_base64_image(s)

        if not_empty(img):
            imgs.append(img)
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return cors_response({
            KEY_OK: false,
            KEY_CODE: 400,
            KEY_MSG: f"Error decoding base64 image: {e}",
            KEY_THROW: e.__class__.__name__,
            # KEY_TRACE: e.__traceback__.__str__
        })

    if is_empty(imgs):
        return cors_response({
            KEY_OK: false,
            KEY_CODE: 400,
            KEY_MSG: 'No file or image parameter found'
        })

    bboxes = []
    with lock:
        for img in imgs:
            results = model(img)  # 进行推理
            if is_empty(results):
                continue

            pose_indexes = []
            for result in results:
                if is_none(result):
                    bboxes.append([])
                    continue

                probs = result.probs  # Probs object for classification outputs
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                result.show()  # display to screen
                result.save(filename="result.jpg")  # save to disk

                conf = boxes.conf
                xywh = boxes.xywh
                cls = boxes.cls

                scores = null if is_none(xywh) else conf.tolist()
                bs = null if is_none(xywh) else xywh.tolist()
                labels = null if is_none(cls) else cls.tolist()
                angles = null if is_none(obb) else obb.tolist()
                # for i in range(boxes.size()):
                #     box = boxes.get(0)
                #     xywh = box.xywh()
                #     bs.append([xywh.x])

                if is_empty(bs):
                    continue

                for i in range(len(bs)):
                    b = bs[i]
                    ind = labels[i] if i < size(labels) else -1
                    label = names[ind] if ind >= 0 and ind < size(names) else '???'
                    if 'person' in label:
                        pose_indexes.append(i)

                    bboxes.append({
                        'id': i,
                        'label': label,
                        'score': scores[i] if i < size(scores) else 0,
                        'angle': angles[i] if i < size(angles) else 0,
                        'color': colors(0) or [255, 0, 0, 0.6],
                        'bbox': b
                    })

            if is_empty(pose_indexes):
                continue

            pose_results = pose_model(img)
            for result in pose_results:
                if is_none(result):
                    continue

                # probs = result.probs  # Probs object for classification outputs
                boxes = result.boxes  # Boxes object for bounding box outputs
                # masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                # obb = result.obb  # Oriented boxes object for OBB outputs
                result.show()  # display to screen
                result.save(filename="result.jpg")  # save to disk

                # conf = boxes.conf
                xy = keypoints.xy
                # cls = boxes.cls

                # scores = null if is_none(xywh) else conf.tolist()
                # bs = null if is_none(xywh) else xywh.tolist()
                # labels = null if is_none(cls) else cls.tolist()
                points = null if is_none(xy) else xy.tolist()
                # angles = null if is_none(obb) else obb.tolist()
                # for i in range(boxes.size()):
                #     box = boxes.get(0)
                #     xywh = box.xywh()
                #     bs.append([xywh.x])

                if is_empty(points):
                    continue

                for i in range(len(points)):
                    p = points[i]
                    ind = pose_indexes[i]
                    bbox = bboxes[ind] or {}
                    bbox['points'] = p

    return cors_response({
        KEY_OK: true,
        KEY_CODE: CODE_SUCCESS,
        KEY_MSG: MSG_SUCCESS,
        'bboxes': bboxes
    })


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True, host='0.0.0.0', port=5000)
