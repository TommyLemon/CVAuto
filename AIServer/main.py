import json
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
    KEY_OK, CODE_SUCCESS, MSG_SUCCESS, KEY_THROW, KEY_TRACE, cur_time_in_millis, get_time_detail, KEY_TIME_DETAIL, \
    is_str
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image

import ssl
# pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org easyocr
import easyocr  # FIXME 不需要或者按以上方式也安装不了依赖的话可以注释掉

ssl._create_default_https_context = ssl._create_unverified_context  # 解决 easyocr 在线下载模型报错 http ssl cert error
reader = easyocr.Reader(['ch_sim', 'en'])  # FIXME 不需要或者安装不了 easyocr 依赖的话可以注释掉

# DEBUG = false
DEBUG = true
MIN_IOU = 0.8

app = Flask(__name__)
CORS(app)

# 设置文件上传的存储路径和允许的文件类型
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

lock = threading.Lock()
# YOLO 模型加载
model = YOLO('yolo11n.pt')  # 使用你选择的模型
pose_model = YOLO('yolo11n-pose.pt')  # 使用你选择的模型
seg_model = YOLO('yolo11n-seg.pt')  # 使用你选择的模型
obb_model = YOLO('yolo11n-obb.pt')  # 使用你选择的模型
names = model.names  # 获取类别名称映射
pose_names = pose_model.names  # 获取类别名称映射
seg_names = seg_model.names  # 获取类别名称映射
obb_names = obb_model.names  # 获取类别名称映射
colors = Colors()

COLOR_RED = [255, 0, 0]
COLOR_GREEN = [0, 255, 0]
COLOR_BLUE = [0, 0, 255]
COLOR_YELLOW = [255, 255, 0]
COLOR_PINK = [255, 0, 255]

KEY_COLOR = 'color'
KEY_ID = 'id'
KEY_SCORE = 'score'
KEY_LABEL = 'label'
KEY_OCR = 'ocr'
KEY_ANGLE = 'angle'
KEY_FILL = 'fill'
KEY_BBOX = 'bbox'
KEY_POLYGON = 'polygon'
KEY_POINT = 'point'
KEY_LINE = 'line'
KEY_BBOXES = 'bboxes'
KEY_POLYGONS = 'polygons'
KEY_POINTS = 'points'
KEY_LINES = 'lines'

# 检查文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_base64_image(base64_string):
    try:
        # 如果是 data:image/jpeg;base64,... 这样的 URI，先去掉前缀
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        # 解码 Base64
        img_data = base64.b64decode(base64_string)

        # 用 PIL 打开
        image = Image.open(BytesIO(img_data)).convert("RGB")  # 加 convert 以避免某些灰度图问题
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
        return url


def compute_iou(b1: list[int], b2: list[int]):
    # 计算两个 bbox（[x, y, w, h, r]）的 IoU

    x1, y1, w1, h1, d1 = b1[0], b1[1], b1[2], b1[3], (0 if size(b1) < 5 else b1[4])
    x2, y2, w2, h2, d2 = b2[0], b2[1], b2[2], b2[3], (0 if size(b2) < 5 else b2[4])

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    iw = xb - xa
    ih = yb - ya

    if iw <= 0 or ih <= 0:
        return 0

    ia = iw * ih
    ua = w1 * h1 + w2 * h2 - ia
    dd = (1 - abs(d1 - d2)/180)

    return (ia / ua) * dd


# 处理推理请求
def cors_response(data):
    host = request.headers.get('Origin') or request.headers.get('Referer') or 'http://localhost'

    rsp = make_response(data if is_str(data) else json.dumps(data), 200)
    # rsp.status = status
    # rsp.status_code = status
    rsp.headers.set('Access-Control-Allow-Origin', host)  # 允许所有域名访问
    rsp.headers.set('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')  # 允许的请求方法
    rsp.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization,Authentication,Cookie,Token,JWT')  # 允许的请求头
    rsp.headers.set('Access-Control-Allow-Credentials', 'true')  # 支持 cookies 或者认证信息
    rsp.headers.set('Access-Control-Max-Age', '3600')  # 预检请求结果缓存时间，单位为秒
    rsp.headers.set('Content-Type', 'application/json;charset=utf-8')  # 固定返回 JSON
    return rsp


@app.route('/detect', methods=['POST', 'OPTIONS'])
def api_detect():
    return predict(is_detect=true, is_pose=false)


@app.route('/pose', methods=['POST', 'OPTIONS'])
def api_pose():
    return predict(is_detect=false, is_pose=true)


@app.route('/segment', methods=['POST', 'OPTIONS'])
def api_segment():
    return predict(is_detect=false, is_pose=false, is_segment=true)


@app.route('/rotate', methods=['POST', 'OPTIONS'])
def api_rotate():
    return predict(is_detect=false, is_pose=false, is_rotate=true)


@app.route('/ocr', methods=['POST', 'OPTIONS'])
def api_ocr():
    return predict(is_detect=false, is_pose=false, is_ocr=true)


@app.route('/predict', methods=['POST', 'OPTIONS'])
def api_predict():
    return predict()


# @cross_origin
def predict(is_detect=true, is_pose: bool = null, is_segment=false, is_ocr: bool = null, is_rotate: bool = null):
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

    min_conf = data.get('conf') or 0.2

    bboxes = []
    polygons = []

    dur = 0
    detect_time_detail = null
    pose_time_detail = null
    seg_time_detail = null
    obb_time_detail = null
    ocr_time_detail = null

    total_start_time = cur_time_in_millis()
    with lock:
        for img in imgs:
            pose_indexes = []
            if is_detect is not False:
                start_time = cur_time_in_millis()
                results = model.predict(img, conf=min_conf)  # 进行推理
                end_time = cur_time_in_millis()
                dur += end_time - start_time
                detect_time_detail = get_time_detail(start_time, end_time)

                for result in results:
                    boxes = null if is_none(result) else result.boxes  # Boxes object for bounding box outputs
                    xyxy = null if is_none(boxes) else boxes.xyxy
                    bs = null if is_none(xyxy) else xyxy.tolist()
                    if is_empty(bs):
                        continue

                    probs = result.probs  # Probs object for classification outputs
                    if DEBUG:
                        result.show()  # display to screen
                        result.save(filename="result_detect.jpg")  # save to disk

                    conf = boxes.conf
                    cls = boxes.cls

                    scores = null if is_none(xyxy) else conf.tolist()
                    labels = null if is_none(cls) else cls.tolist()

                    for i in range(len(bs)):
                        c = scores[i] if i < size(scores) else 0
                        if c < min_conf:
                            continue

                        b = bs[i]
                        ind = labels[i] if i < size(labels) else -1
                        label = names[int(ind)] if ind >= 0 and int(ind) < size(names) else '???'
                        if 'person' in label:
                            pose_indexes.append(i)

                        bboxes.append({
                            KEY_ID: i + 1,
                            KEY_LABEL: label,
                            KEY_SCORE: c,
                            KEY_COLOR: colors(ind%colors.n) or [255, 0, 0, 0.6],
                            KEY_BBOX: [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                        })

            if is_pose or (is_pose is None and is_detect and not_empty(pose_indexes)):
                start_time = cur_time_in_millis()
                pose_results = pose_model.predict(img, conf=min_conf)
                end_time = cur_time_in_millis()
                dur += end_time - start_time
                pose_time_detail = get_time_detail(start_time, end_time)
                if not_none(pose_results):
                    for result in pose_results:
                        keypoints = null if is_none(result) else result.keypoints  # Keypoints object for pose outputs
                        xy = null if is_none(keypoints) else keypoints.xy
                        points = null if is_none(xy) else xy.tolist()
                        if is_empty(points):
                            continue

                        # probs = result.probs  # Probs object for classification outputs
                        boxes = result.boxes  # Boxes object for bounding box outputs
                        if DEBUG:
                            result.show()  # display to screen
                            result.save(filename="result_pose.jpg")  # save to disk

                        conf = boxes.conf
                        cls = boxes.cls
                        xyxy = boxes.xyxy

                        scores = null if is_none(conf) else conf.tolist()
                        labels = null if is_none(cls) else cls.tolist()
                        bs = null if is_none(xyxy) else xyxy.tolist()

                        for i in range(len(points)):
                            c = scores[i] if i < size(scores) else 0
                            if c < min_conf:
                                continue

                            ps = points[i]
                            # 不准 ind = (pose_indexes[i] if i < size(pose_indexes) else -1) if is_pose is None else (labels[i] if i < size(labels) else -1)
                            ind = labels[i] if i < size(labels) else -1
                            label = pose_names[int(ind)] if ind >= 0 and int(ind) < size(pose_names) else '???'

                            if size(ps) == 5:  # 人脸
                                p0 = ps[0]
                                p1 = ps[1]
                                p2 = ps[2]
                                p3 = ps[3]
                                p4 = ps[4]
                                lines = [
                                    [p0[0], p0[1], p1[0], p1[1]],
                                    [p1[0], p1[1], p2[0], p2[1]],
                                    [p2[0], p2[1], p3[0], p3[1]],
                                    [p3[0], p3[1], p4[0], p4[1]],
                                    [p4[0], p4[1], p2[0], p2[1]],
                                    [p2[0], p2[1], p0[0], p0[1]]
                                ]
                            elif size(ps) == 17:  # 人体姿态
                                p0 = ps[0]  # 鼻子（nose）
                                p1 = ps[1]  # 左眼（left_eye）
                                p2 = ps[2]  # 右眼（right_eye）
                                p3 = ps[3]  # 左耳（left_ear）
                                p4 = ps[4]  # 右耳（right_ear）
                                p5 = ps[5]  # 左肩（left_shoulder）
                                p6 = ps[6]  # 右肩（right_shoulder）
                                p7 = ps[7]  # 左肘（left_elbow）
                                p8 = ps[8]  # 右肘（right_elbow）
                                p9 = ps[9]  # 左腕（left_wrist）
                                p10 = ps[10]  # 右腕（right_wrist）
                                p11 = ps[11]  # 左髋（left_hip）
                                p12 = ps[12]  # 右髋（right_hip）
                                p13 = ps[13]  # 左膝（left_knee）
                                p14 = ps[14]  # 右膝（right_knee）
                                p15 = ps[15]  # 左踝（left_ankle）
                                p16 = ps[16]  # 右踝（right_ankle）

                                lines = [
                                    # 头部
                                    {
                                        KEY_LINE: [p0[0], p0[1], p1[0], p1[1]],
                                        KEY_COLOR: COLOR_GREEN
                                    }, {
                                        KEY_LINE: [p0[0], p0[1], p2[0], p2[1]],
                                        KEY_COLOR: COLOR_GREEN
                                    }, {
                                        KEY_LINE: [p1[0], p1[1], p3[0], p3[1]],
                                        KEY_COLOR: COLOR_GREEN
                                    }, {
                                        KEY_LINE: [p2[0], p2[1], p4[0], p4[1]],
                                        KEY_COLOR: COLOR_GREEN
                                    },

                                    # 头肩
                                    {
                                        KEY_LINE: [p3[0], p3[1], p5[0], p5[1]],
                                        KEY_COLOR: COLOR_GREEN
                                    }, {
                                        KEY_LINE: [p4[0], p4[1], p6[0], p6[1]],
                                        KEY_COLOR: COLOR_GREEN
                                    },


                                    # 肩膀
                                    {
                                        KEY_LINE: [p5[0], p5[1], p6[0], p6[1]],
                                        KEY_COLOR: COLOR_BLUE
                                    },

                                    # 手臂
                                    {
                                        KEY_LINE: [p5[0], p5[1], p7[0], p7[1]],
                                        KEY_COLOR: COLOR_BLUE
                                    }, {
                                        KEY_LINE: [p7[0], p7[1], p9[0], p9[1]],
                                        KEY_COLOR: COLOR_BLUE
                                    }, {
                                        KEY_LINE: [p6[0], p6[1], p8[0], p8[1]],
                                        KEY_COLOR: COLOR_BLUE
                                    }, {
                                        KEY_LINE: [p8[0], p8[1], p10[0], p10[1]],
                                        KEY_COLOR: COLOR_BLUE
                                    },

                                    # 躯干
                                    {
                                        KEY_LINE: [p5[0], p5[1], p11[0], p11[1]],
                                        KEY_COLOR: COLOR_PINK
                                    }, {
                                        KEY_LINE: [p6[0], p6[1], p12[0], p12[1]],
                                        KEY_COLOR: COLOR_PINK
                                    },

                                    # 左腿
                                    {
                                        KEY_LINE: [p11[0], p11[1], p12[0], p12[1]],
                                        KEY_COLOR: COLOR_YELLOW
                                    }, {
                                        KEY_LINE: [p11[0], p11[1], p13[0], p13[1]],
                                        KEY_COLOR: COLOR_YELLOW
                                    }, {
                                        KEY_LINE: [p13[0], p13[1], p15[0], p15[1]],
                                        KEY_COLOR: COLOR_YELLOW
                                    },

                                    # 右腿
                                    {
                                        KEY_LINE: [p12[0], p12[1], p14[0], p14[1]],
                                        KEY_COLOR: COLOR_YELLOW
                                    }, {
                                        KEY_LINE: [p14[0], p14[1], p16[0], p16[1]],
                                        KEY_COLOR: COLOR_YELLOW
                                    }
                                ]

                            b = null if i >= size(bs) else bs[i]
                            if true:  # ind is None or int(ind) >= size(bboxes):
                                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                                pb = null if is_empty(b) else [int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)]
                                repeat = -1
                                if not_empty(pb):
                                    for j in range(size(bboxes)):
                                        det_bbox = bboxes[j]
                                        db = null if label != det_bbox.get(KEY_LABEL) else det_bbox.get(KEY_BBOX)
                                        if not_empty(db) and compute_iou(db, pb) > MIN_IOU:
                                            repeat = j
                                            break

                                if repeat >= 0:
                                    del bboxes[repeat]

                                bboxes.insert(0, {
                                    KEY_ID: i + 1,
                                    KEY_LABEL: label,
                                    KEY_SCORE: c,
                                    KEY_COLOR: colors(ind%colors.n) or [0, 255, 0, 0.6],
                                    KEY_BBOX: pb,
                                    KEY_POINTS: ps,
                                    KEY_LINES: lines
                                })


            if is_segment or (is_segment is None and is_detect and size(bboxes) < 10):
                start_time = cur_time_in_millis()
                seg_results = seg_model.predict(img, conf=min_conf)
                end_time = cur_time_in_millis()
                dur += end_time - start_time
                seg_time_detail = get_time_detail(start_time, end_time)

                if not_none(seg_results):
                    for result in seg_results:
                        masks = null if is_none(result) else result.masks  # Masks object for segmentation masks outputs
                        xy = null if is_none(masks) else masks.xy
                        pointss = null if is_none(xy) else [p.tolist() for p in xy]
                        if is_empty(pointss):
                            continue

                        boxes = result.boxes  # Boxes object for bounding box outputs
                        if DEBUG:
                            result.show()  # display to screen
                            result.save(filename="result_seg.jpg")  # save to disk

                        conf = boxes.conf
                        cls = boxes.cls
                        xyxy = boxes.xyxy

                        scores = null if is_none(conf) else conf.tolist()
                        labels = null if is_none(cls) else cls.tolist()
                        bs = null if is_none(xyxy) else xyxy.tolist()

                        for i in range(len(pointss)):
                            c = scores[i] if i < size(scores) else 0
                            points = null if c < min_conf else pointss[i]
                            b = null if size(points) < 3 or i >= size(bs) else bs[i]
                            if is_empty(b):
                                continue

                            ind = labels[i] if i < size(labels) else -1
                            label = seg_names[int(ind)] if ind >= 0 and int(ind) < size(seg_names) else '???'

                            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                            sb = null if is_empty(b) else [int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)]
                            repeat = -1
                            if not_empty(sb):
                                for j in range(size(bboxes)):
                                    det_bbox = bboxes[j]
                                    db = null if label != det_bbox.get(KEY_LABEL) else det_bbox.get(KEY_BBOX)
                                    if not_empty(db) and compute_iou(db, sb) > MIN_IOU:
                                        repeat = j
                                        break

                            if repeat >= 0:
                                del bboxes[repeat]

                            bboxes.insert(0, {
                                KEY_ID: i + 1,
                                KEY_LABEL: label,
                                KEY_SCORE: c,
                                KEY_COLOR: colors(ind % colors.n) or [0, 0, 255, 0.6],
                                KEY_BBOX: sb
                            })
                            polygons.insert(0, {
                                KEY_ID: i + 1,
                                KEY_LABEL: label,
                                KEY_SCORE: c,
                                KEY_FILL: true,
                                KEY_COLOR: colors(ind%colors.n) or [0, 0, 255, 0.6],
                                # KEY_BBOX: sb,
                                KEY_POINTS: points
                            })

            if is_rotate or (is_rotate is None and is_ocr):
                start_time = cur_time_in_millis()
                obb_results = obb_model.predict(img, conf=min_conf)
                end_time = cur_time_in_millis()
                dur += end_time - start_time
                obb_time_detail = get_time_detail(start_time, end_time)
                if not_none(obb_results):
                    for result in obb_results:
                        obb = null if is_none(result) else result.obb  # Oriented boxes object for OBB outputs
                        xyxyxyxy = null if is_none(obb) else obb.xyxyxyxy
                        obs = null if is_none(xyxyxyxy) else xyxyxyxy.tolist()
                        if is_empty(obs):
                            continue

                        if DEBUG:
                            result.show()  # display to screen
                            result.save(filename="result_obb.jpg")  # save to disk

                        conf = obb.conf
                        cls = obb.cls
                        xywhr = obb.xywhr

                        scores = null if is_none(conf) else conf.tolist()
                        labels = null if is_none(cls) else cls.tolist()
                        angles = null if is_none(xywhr) else xywhr.tolist()

                        for i in range(len(obs)):
                            c = scores[i] if i < size(scores) else 0
                            if c < min_conf:
                                continue

                            ob = obs[i]
                            x1, y1 = np.min(ob, axis=0)
                            x2, y2 = np.max(ob, axis=0)

                            ind = labels[i] if i < size(labels) else -1
                            label = obb_names[int(ind)] if ind >= 0 and int(ind) < size(obb_names) else ''

                            bboxes.append({
                                KEY_ID: i + 1,
                                KEY_LABEL: label,
                                KEY_SCORE: c,
                                KEY_ANGLE: null if i >= size(angles) else 2*angles[i][4],
                                KEY_COLOR: colors(ind % colors.n) or [100, 100, 0, 0.6],
                                KEY_BBOX: [int(x1), int(y1), int(x2) - int(x1), int(y2)- int(y1)]
                            })
                            polygons.append({
                                KEY_ID: i + 1,
                                KEY_LABEL: label,
                                KEY_SCORE: c,
                                KEY_FILL: false,
                                KEY_COLOR: colors(ind % colors.n) or [100, 100, 0, 0.6],
                                KEY_POINTS: [[int(obi[0]), int(obi[1])] for obi in ob]
                            })

            if is_ocr or (is_ocr is None and is_rotate):
                start_time = cur_time_in_millis()
                ocr_results = reader.readtext(img, text_threshold=min_conf)
                end_time = cur_time_in_millis()
                dur += end_time - start_time
                ocr_time_detail = get_time_detail(start_time, end_time)
                if not_none(ocr_results):
                    i = 0
                    for bbox, text, conf in ocr_results:
                        if is_empty(bbox):
                            continue

                        # 转为 [x1, y1, x2, y2]
                        x1, y1 = np.min(bbox, axis=0)
                        x2, y2 = np.max(bbox, axis=0)

                        bboxes.append({
                            KEY_ID: i + 1,
                            # KEY_LABEL: label,
                            KEY_OCR: text,
                            KEY_SCORE: float(conf),
                            # KEY_ANGLE: 0, # TODO 根据长边顶点与中心角度差算旋转角度？还是先对齐水平的长短
                            KEY_COLOR: [0, 100, 100, 0.6],
                            KEY_BBOX: [int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)]
                        })

                        polygons.append({
                            KEY_ID: i + 1,
                            # KEY_LABEL: label,
                            # KEY_OCR: text,
                            KEY_FILL: false,
                            KEY_SCORE: float(conf),
                            KEY_COLOR: [0, 100, 100, 0.6],
                            KEY_POINTS: [[int(item[0]), int(item[1])] for item in bbox],
                        })

                        i += 1

    total_end_time = cur_time_in_millis()
    total_dur = total_end_time - total_start_time
    total_time_detail = get_time_detail(total_start_time, total_end_time)

    rsp = {
        KEY_BBOXES: bboxes,
        KEY_POLYGONS: polygons,
        KEY_OK: true,
        KEY_CODE: CODE_SUCCESS,
        KEY_MSG: MSG_SUCCESS,
        KEY_TIME_DETAIL + '|parse|sql|model': total_time_detail + f'|{total_dur - dur}|0|{dur}'
    }
    if not_empty(detect_time_detail):
        rsp['det-' + KEY_TIME_DETAIL] = detect_time_detail
    if not_empty(pose_time_detail):
        rsp['pose-' + KEY_TIME_DETAIL] = pose_time_detail
    if not_empty(seg_time_detail):
        rsp['seg-' + KEY_TIME_DETAIL] = seg_time_detail
    if not_empty(obb_time_detail):
        rsp['obb-' + KEY_TIME_DETAIL] = obb_time_detail
    if not_empty(ocr_time_detail):
        rsp['ocr-' + KEY_TIME_DETAIL] = ocr_time_detail

    return cors_response(rsp)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=DEBUG, host='0.0.0.0', port=5000)
