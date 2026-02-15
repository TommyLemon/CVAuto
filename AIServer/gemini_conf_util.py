"""
Gemini 动态置信度阈值计算工具 (gemini_conf_util.py)

该模块提供了一个核心函数 `calculate_dynamic_thresholds`，用于根据摄像头的
物理参数、场景几何以及时序信息，为YOLO等检测器输出的每个边界框计算一个
动态的、最合适的置信度阈值。

策略优先级:
1. 透视变换模型 (Homography): 最精确，需要4点标定。
2. 几何模型 (Geometric): 次精确，需要相机物理参数(height, vfov等)。
3. 启发式模型 (Heuristic): 基于像素位置(X, Y)和大小的经验规则。
"""

import threading
import math

try:
    import numpy as np
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("警告: numpy 或 opencv-python 未安装。透视变换(Homography)功能将不可用。")

# --- 模块级全局变量，用于跨请求/帧的状态管理 ---

# 全局视频状态缓存
# 结构: { "video_id_1": { "frame_index": 123, "tracked_objects": [...] }, ... }
VIDEO_STATS_CACHE = {}
# 用于保证缓存读写线程安全的锁
STATS_LOCK = threading.Lock()


# --- 内部辅助函数 ---

def _get_center_distance(box1, box2):
    """(辅助函数) 计算两个边界框中心的欧氏距离"""
    center1_x = box1[0] + (box1[2] - box1[0]) / 2
    center1_y = box1[1] + (box1[3] - box1[1]) / 2
    center2_x = box2[0] + (box2[2] - box2[0]) / 2
    center2_y = box2[1] + (box2[3] - box2[1]) / 2
    return math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)


# --- 透视变换(Homography)相关函数 ---

def _calculate_homography(pixel_points, world_points):
    """
    (辅助函数) 根据四对匹配点计算单应性矩阵。
    需要 opencv-python。
    """
    if not OPENCV_AVAILABLE:
        return None

    if len(pixel_points) != 4 or len(world_points) != 4:
        return None

    try:
        src_pts = np.array(pixel_points, dtype=np.float32)
        dst_pts = np.array(world_points, dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    except Exception as e:
        print(f"[gemini_conf_util] 错误：计算单应性矩阵失败: {e}")
        return None


def _get_distance_from_pixel(u, v, homography_matrix, camera_height):
    """
    (辅助函数) 使用单应性矩阵和相机高度，计算像素点对应的真实世界距离。
    """
    if homography_matrix is None or not OPENCV_AVAILABLE:
        return float('inf')

    pixel_coords = np.array([[[u, v]]], dtype=np.float32)
    world_coords_homogeneous = cv2.perspectiveTransform(pixel_coords, homography_matrix)

    wx = world_coords_homogeneous[0][0][0]
    wy = world_coords_homogeneous[0][0][1]

    distance = math.sqrt(wx ** 2 + wy ** 2 + camera_height ** 2)
    return distance


# --- 时序平滑 (EMA) 相关函数 ---

def _get_smoothed_objects(all_boxes, video_id, frame_index, config):
    """
    (辅助函数) 执行轻量级追踪和EMA平滑，用于 'enable_size_adjustment'。
    """
    newly_tracked_objects = []

    smoothing_factor = config.get('smoothing_factor', 0.2)
    tracking_dist_thresh = config.get('tracking_dist_thresh', 50)

    with STATS_LOCK:
        last_frame_data = VIDEO_STATS_CACHE.get(video_id, {})
        last_tracked_objects = last_frame_data.get('tracked_objects', [])
        last_frame_index = last_frame_data.get('frame_index', -1)

        if frame_index - last_frame_index > 10:  # 跳帧太多则重置
            last_tracked_objects = []

        unmatched_detections = list(all_boxes)

        for tracked_obj in last_tracked_objects:
            best_match, best_match_idx, best_match_dist = None, -1, float('inf')
            for i, det_box in enumerate(unmatched_detections):
                dist = _get_center_distance(tracked_obj['box'], det_box)
                if dist < tracking_dist_thresh and dist < best_match_dist:
                    best_match, best_match_idx, best_match_dist = det_box, i, dist

            if best_match:
                matched_det_box = unmatched_detections.pop(best_match_idx)
                current_height = matched_det_box[3] - matched_det_box[1]
                smoothed_height = (smoothing_factor * current_height) + (1 - smoothing_factor) * tracked_obj[
                    'smoothed_height']

                tracked_obj['box'] = matched_det_box
                tracked_obj['smoothed_height'] = smoothed_height
                newly_tracked_objects.append(tracked_obj)

        for new_det_box in unmatched_detections:
            newly_tracked_objects.append({
                'box': new_det_box,
                'smoothed_height': new_det_box[3] - new_det_box[1],  # 初始高度
            })

        VIDEO_STATS_CACHE[video_id] = {'frame_index': frame_index, 'tracked_objects': newly_tracked_objects}

    return newly_tracked_objects


# --- 对外暴露的主函数 ---

def calculate_dynamic_thresholds(all_boxes, config, image_shape):
    """
    根据配置和时序信息，为当前帧的每个检测框计算动态置信度阈值。

    Args:
        all_boxes (list): 当前帧检测出的所有原始边界框列表 (xyxy格式)。
        config (dict): 包含所有控制参数的字典。
        image_shape (tuple): 图像的形状 (height, width, ...)。

    Returns:
        dict: 一个映射字典，键为边界框的索引(0, 1, 2, ...)，值为其对应的动态置信度阈值。
    """

    # --- 1. 解析通用参数 ---
    min_conf = config.get('conf', 0.2)
    max_conf = config.get('max_conf', 0.8)
    img_height = image_shape[0] if image_shape and len(image_shape) > 0 else 0
    img_width = image_shape[1] if image_shape and len(image_shape) > 1 else 0

    threshold_map = {}

    # --- 2. 策略一 (最高优先级): 透视变换模型 (Homography) ---
    enable_perspective = config.get('enable_perspective_adjustment', False)
    if enable_perspective and OPENCV_AVAILABLE:
        pixel_points = config.get('pixel_points', [])
        world_points = config.get('world_points', [])
        height = config.get('height', 0.0)

        homography_matrix = _calculate_homography(pixel_points, world_points)

        if homography_matrix is not None and height > 0:
            distance_near = config.get('distance_near', 5.0)  # 用于插值的最近物理距离(米)
            distance_far = config.get('distance_far', 50.0)  # 用于插值的最远物理距离(米)
            dist_range = distance_far - distance_near

            for i, b in enumerate(all_boxes):
                u, v = (b[0] + b[2]) / 2, b[3]  # 取框底部中心点
                object_distance = _get_distance_from_pixel(u, v, homography_matrix, height)

                perspective_thresh = min_conf
                if dist_range > 0:
                    dist_factor = (object_distance - distance_near) / dist_range
                    dist_factor = max(0.0, min(1.0, dist_factor))  # 0=近, 1=远
                    perspective_thresh = max_conf - (max_conf - min_conf) * dist_factor

                threshold_map[i] = perspective_thresh

            # (透视模型不与其他模型叠加，它已足够精确)
            return threshold_map

    # --- 3. 策略二 (第二优先级): 物理几何模型 ---
    enable_geometric = config.get('enable_geometric_adjustment', False)
    if enable_geometric and img_height > 0:
        height = config.get('height', 0.0)
        width = config.get('width', 0.0)
        camera_vfov = config.get('camera_vfov', 0.0)
        width_bottom = config.get('width_bottom', 0.0)

        tilt_angle_rad, vfov_rad = 0, 0

        if height > 0 and width > 0:
            if camera_vfov > 0:
                vfov_rad = math.radians(camera_vfov)
                tilt_angle_rad = math.atan(width / height)
            elif width_bottom > width:
                angle_to_center = math.atan(width / height)
                angle_to_bottom = math.atan(width_bottom / height)
                half_vfov_rad = angle_to_bottom - angle_to_center
                if half_vfov_rad > 0:
                    vfov_rad = half_vfov_rad * 2
                    tilt_angle_rad = angle_to_center

        if tilt_angle_rad > 0 and vfov_rad > 0:
            # (此处的距离插值逻辑可以复用策略一的 distance_near/distance_far)
            # (为保持代码独立性，这里重新计算)
            try:
                angle_to_bottom = tilt_angle_rad + (vfov_rad / 2)
                dist_o_near = height * math.tan(angle_to_bottom)
                distance_near = math.sqrt(height ** 2 + dist_o_near ** 2)
                dist_o_center = height * math.tan(tilt_angle_rad)
                distance_far = math.sqrt(height ** 2 + dist_o_center ** 2)
                dist_range = distance_far - distance_near
            except ValueError:
                dist_range = -1

            for i, b in enumerate(all_boxes):
                y_pixel_pos = b[3]  # 取框底部
                angle_offset_rad = ((img_height / 2) - y_pixel_pos) / (img_height / 2) * (vfov_rad / 2)
                total_angle_rad = tilt_angle_rad - angle_offset_rad

                geometric_thresh = min_conf
                if dist_range > 0 and total_angle_rad > 0 and total_angle_rad < math.pi / 2:
                    dist_from_o = height * math.tan(total_angle_rad)
                    object_distance = math.sqrt(height ** 2 + dist_from_o ** 2)
                    dist_factor = (object_distance - distance_near) / dist_range
                    dist_factor = max(0.0, min(1.0, dist_factor))  # 0=近, 1=远
                    geometric_thresh = max_conf - (max_conf - min_conf) * dist_factor

                threshold_map[i] = geometric_thresh

            # (几何模型也不与其他启发式模型叠加)
            return threshold_map

    # --- 4. 策略三 (回退): 启发式模型 (Y坐标, 尺寸, X坐标) ---

    # 解析启发式模型所需参数
    enable_distance = config.get('enable_distance_adjustment', False)
    enable_size = config.get('enable_size_adjustment', False)
    enable_horizontal = config.get('enable_horizontal_adjustment', False)

    y_pos_strategy = config.get('y_pos_strategy', 'bottom')
    avg_height_far = config.get('avg_height_far', 0.0)
    avg_height_near = config.get('avg_height_near', 0.0)
    edge_multiplier = config.get('edge_confidence_multiplier', 1.0)

    video_id = config.get('video_id')
    frame_index = config.get('frame_index', 0)

    # --- 4a. 如果启用尺寸调整，则预先进行时序平滑 ---
    smoothed_objects = []
    if enable_size and video_id:
        smoothed_objects = _get_smoothed_objects(all_boxes, video_id, frame_index, config)

    # --- 4b. 遍历所有框，计算启发式阈值 ---
    for i, b in enumerate(all_boxes):
        distance_thresh = min_conf
        size_thresh = min_conf

        # (Y坐标距离)
        if enable_distance and img_height > 0:
            y_pos = b[3] if y_pos_strategy == 'bottom' else (b[1] + b[3]) / 2
            center_line_y = img_height / 2.0
            dist_factor = 1.0
            if y_pos > center_line_y:
                denominator = img_height - center_line_y
                if denominator > 0:
                    dist_factor = (img_height - y_pos) / denominator
            dist_factor = max(0.0, min(1.0, dist_factor))  # 0=近, 1=远
            distance_thresh = max_conf + (min_conf - max_conf) * dist_factor

        # (尺寸大小)
        if enable_size:
            box_height = b[3] - b[1]
            if video_id:  # 查找平滑后的高度
                for obj in smoothed_objects:
                    if obj['box'] == b:
                        box_height = obj['smoothed_height']
                        break

            size_range = avg_height_near - avg_height_far
            if size_range > 0:
                size_factor = (box_height - avg_height_far) / size_range
                size_factor = max(0.0, min(1.0, size_factor))  # 0=小, 1=大
                size_thresh = min_conf + (max_conf - min_conf) * size_factor

        # (组合) 取Y坐标和尺寸策略中更严格（大）的那个
        base_threshold = max(distance_thresh, size_thresh)

        # (X坐标调整)
        if enable_horizontal and img_width > 0:
            image_center_x = img_width / 2.0
            object_center_x = (b[0] + b[2]) / 2
            horizontal_factor = abs(object_center_x - image_center_x) / image_center_x  # 0=中心, 1=边缘
            threshold_multiplier = 1.0 - (1.0 - edge_multiplier) * horizontal_factor
            base_threshold = base_threshold * threshold_multiplier

        threshold_map[i] = base_threshold

    return threshold_map