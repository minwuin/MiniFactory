import cv2
import numpy as np
from ultralytics import YOLO

def merge_lines(lines, rho_threshold=20, theta_threshold=np.pi/15):
    """
    1픽셀 뼈대를 기반으로 선을 추출하므로, 미세한 노이즈만 가볍게 병합합니다.
    """
    if lines is None:
        return []
    
    clusters = []
    for line in lines:
        rho, theta = line[0]
        
        found_cluster = False
        for cluster in clusters:
            c_rho, c_theta = cluster['center']
            
            d_theta = abs(theta - c_theta)
            if d_theta > np.pi / 2:
                d_theta = np.pi - d_theta
                d_rho = abs(rho + c_rho) 
            else:
                d_rho = abs(rho - c_rho)
                
            if d_theta < theta_threshold and d_rho < rho_threshold:
                cluster['lines'].append((rho, theta))
                cluster['center'] = (
                    np.mean([l[0] for l in cluster['lines']]),
                    np.mean([l[1] for l in cluster['lines']])
                )
                found_cluster = True
                break
                
        if not found_cluster:
            clusters.append({'center': (rho, theta), 'lines': [(rho, theta)]})
            
    return [c['center'] for c in clusters]

def segments_to_rho_theta(segments):
    """확률적 허프 변환의 선분 좌표(x1, y1, x2, y2)를 기존 (rho, theta) 포맷으로 변환"""
    if segments is None:
        return None
    
    raw_lines = []
    for segment in segments:
        x1, y1, x2, y2 = segment[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            continue
        
        # 법선 벡터의 각도 (theta) 계산
        theta = np.arctan2(dx, -dy)
        if theta < 0:
            theta += np.pi
            
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        raw_lines.append([[rho, theta]])
        
    return raw_lines if raw_lines else None

class VisionManager:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)
        self.prev_lines = []
        self.alpha = 0.3
        self.class_mapping = {0: "ground", 1: "car", 2: "A", 3: "B", 4: "C", 5: "D", 6: "E", 7: "F", 8: "G"}
        
        # --- 확률적 허프 변환 및 필터링 파라미터 ---
        self.hough_thresh = 40       # 픽셀 점 개수 임계값
        self.min_line_length = 40    # [핵심] 이 길이보다 짧은 선분(교차점 노이즈 등)은 무시
        self.max_line_gap = 50       # 이 픽셀 이내로 끊어진 선분은 하나의 선으로 연결
        self.ema_rho_limit = 50

    def get_skeleton(self, img):
        skeleton = np.zeros(img.shape, np.uint8)
        temp_img = img.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(temp_img, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(temp_img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            temp_img = eroded.copy()
            if cv2.countNonZero(temp_img) == 0:
                break
        return skeleton

    def process_frame(self, frame, locked_lines=None):
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width // 2, height // 2))

        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()

        paper_mask = np.zeros((height // 2, width // 2), dtype=np.uint8)
        object_centers = []

        if results[0].masks is not None:
            for i, mask_xy in enumerate(results[0].masks.xy):
                class_id = int(results[0].boxes.cls[i])
                if class_id == 0:
                    cv2.fillPoly(paper_mask, np.int32([mask_xy]), 255)
            
            for i, mask_xy in enumerate(results[0].masks.xy):
                class_id = int(results[0].boxes.cls[i])
                if class_id != 0:
                    cv2.fillPoly(paper_mask, np.int32([mask_xy]), 0)
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    if class_id != 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        object_centers.append({
                            "class_name": self.class_mapping.get(class_id, "unknown"),
                            # 정중앙 중심점으로 변경 및 키 이름을 center로 수정
                            "center": (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        })

        # --- [추가됨] 바닥 테두리 경계선 노이즈 제거 (침식 연산) ---
        # 흰 바닥 마스크(paper_mask)를 안쪽으로 살짝 깎아내어 바깥쪽 배경과의 충돌을 막습니다.
        margin_size = 8  # 깎아낼 두께 (경계선이 계속 잡히면 20, 25로 늘리십시오)
        erosion_kernel = np.ones((margin_size, margin_size), np.uint8)
        paper_mask = cv2.erode(paper_mask, erosion_kernel, iterations=1)

        # HSV 기반 검은 테이프 추출 (적용 완료된 최적값 유지)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([179, 255, 78])
        
        binary_tape = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        tape_mask = cv2.bitwise_and(binary_tape, binary_tape, mask=paper_mask)
        skeleton_edges = self.get_skeleton(tape_mask)

        if locked_lines is not None:
            current_lines = locked_lines
        else:
            # --- [변경됨] 확률적 허프 변환 적용 ---
            segments = cv2.HoughLinesP(
                skeleton_edges, 1, np.pi/180, threshold=self.hough_thresh,
                minLineLength=self.min_line_length, maxLineGap=self.max_line_gap
            )
            
            # 선분(Segments)을 기존의 (rho, theta) 무한선 포맷으로 변환
            raw_lines = segments_to_rho_theta(segments)
            
            # 각도 기반 대각선 노이즈 필터링 (수직, 수평 보장)
            filtered_lines = []
            if raw_lines is not None:
                angle_margin = np.pi / 8  # 허용 오차: 약 22.5도
                
                for line in raw_lines:
                    rho, theta = line[0]
                    is_vertical = (theta < angle_margin) or (theta > np.pi - angle_margin)
                    is_horizontal = abs(theta - np.pi/2) < angle_margin
                    
                    if is_vertical or is_horizontal:
                        filtered_lines.append(line)
            
            merged_lines = merge_lines(filtered_lines if filtered_lines else None)

            smoothed_lines = []
            if merged_lines is not None:
                for rho, theta in merged_lines:
                    best_match = None
                    min_diff = float('inf')

                    for prho, ptheta in self.prev_lines:
                        d_theta = abs(theta - ptheta)
                        if d_theta > np.pi / 2:
                            d_theta = np.pi - d_theta
                        d_rho = abs(rho - prho)

                        if d_theta < (np.pi / 10) and d_rho < self.ema_rho_limit:
                            diff = d_rho + d_theta * 100
                            if diff < min_diff:
                                min_diff = diff
                                best_match = (prho, ptheta)

                    if best_match:
                        prho, ptheta = best_match
                        
                        diff_theta = theta - ptheta
                        if diff_theta > np.pi / 2:
                            theta -= np.pi
                            rho = -rho
                        elif diff_theta < -np.pi / 2:
                            theta += np.pi
                            rho = -rho

                        s_rho = self.alpha * rho + (1 - self.alpha) * prho
                        s_theta = self.alpha * theta + (1 - self.alpha) * ptheta

                        if s_theta < 0:
                            s_theta += np.pi
                            s_rho = -s_rho
                        elif s_theta >= np.pi:
                            s_theta -= np.pi
                            s_rho = -s_rho

                        smoothed_lines.append((s_rho, s_theta))
                    else:
                        smoothed_lines.append((rho, theta))

            self.prev_lines = smoothed_lines.copy()
            current_lines = smoothed_lines

        line_img = np.zeros_like(annotated_frame)

        for rho, theta in current_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * (a))
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * (a))
            
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        masked_lines = cv2.bitwise_and(line_img, line_img, mask=paper_mask)
        indices = np.where(masked_lines[:, :, 1] == 255)
        annotated_frame[indices[0], indices[1], :] = [0, 255, 0]

        return annotated_frame, paper_mask, skeleton_edges, current_lines, object_centers