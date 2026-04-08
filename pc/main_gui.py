import sys
import cv2
import numpy as np
import time
import socket
import json
from collections import Counter

from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                               QHBoxLayout, QWidget, QPushButton, QTextEdit, 
                               QComboBox, QMessageBox)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import QThread, Signal, Qt

# 분리된 폴더에서 클래스 호출
from vision.vision_manager import VisionManager
from navigation.pathfinder import DijkstraPathfinder

def build_grid_map(locked_lines, object_centers):
    if locked_lines is None:
        return {}, 0, 0, {}

    v_lines, h_lines = [], []
    for rho, theta in locked_lines:
        angle_margin = np.pi / 8
        if (theta < angle_margin) or (theta > np.pi - angle_margin):
            v_lines.append((rho, theta))
        elif abs(theta - np.pi/2) < angle_margin:
            h_lines.append((rho, theta))
            
    def get_x_intercept(line, y=240):
        if np.cos(theta) == 0: return line[0]
        return (line[0] - y * np.sin(line[1])) / np.cos(line[1])
        
    def get_y_intercept(line, x=320):
        if np.sin(theta) == 0: return line[0]
        return (line[0] - x * np.cos(line[1])) / np.sin(line[1])
        
    v_lines.sort(key=lambda l: get_x_intercept(l, 240))
    h_lines.sort(key=lambda l: get_y_intercept(l, 320), reverse=True) 
    
    v_xs = [get_x_intercept(l, 240) for l in v_lines]
    h_ys = [get_y_intercept(l, 320) for l in h_lines]
    avg_w = np.mean(np.diff(v_xs)) if len(v_xs) > 1 else 100
    avg_h = np.mean(np.abs(np.diff(h_ys))) if len(h_ys) > 1 else 100

    max_x, max_y = len(v_lines) + 1, len(h_lines) + 1
    cell_centers = {}
    
    for r in range(1, max_y + 1):
        for c in range(1, max_x + 1):
            if c == 1: cx = v_xs[0] - avg_w / 2 if v_xs else 320
            elif c == max_x: cx = v_xs[-1] + avg_w / 2 if v_xs else 320
            else: cx = (v_xs[c-2] + v_xs[c-1]) / 2
                
            if r == 1: cy = h_ys[0] + avg_h / 2 if h_ys else 240
            elif r == max_y: cy = h_ys[-1] - avg_h / 2 if h_ys else 240
            else: cy = (h_ys[r-2] + h_ys[r-1]) / 2

            cell_centers[(c, r)] = (int(cx), int(cy))

    grid_map = {}
    for obj in object_centers:
        cls_name = obj.get('class_name', 'unknown')
        cx, cy = obj.get('center', (0, 0))
        
        col_idx = 1
        for v_line in v_lines:
            if cx > get_x_intercept(v_line, cy): col_idx += 1
            else: break
                
        row_idx = 1
        for h_line in h_lines:
            if cy < get_y_intercept(h_line, cx): row_idx += 1
            else: break
                
        coord = (col_idx, row_idx)
        if coord not in grid_map:
            grid_map[coord] = []
        grid_map[coord].append(cls_name)
        
    return grid_map, max_x, max_y, cell_centers

# --- [신규 추가] 네트워크 통신 스레드 ---
class ServerThread(QThread):
    """PC에 소켓 서버를 열고 라즈베리파이의 접속과 메시지를 백그라운드에서 처리합니다."""
    client_connected_signal = Signal(str)
    client_disconnected_signal = Signal()
    message_received_signal = Signal(str)

    def __init__(self, port=8081):
        super().__init__()
        self.port = port
        self._run_flag = True
        self.client_socket = None
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0) # 안전한 종료를 위한 타임아웃

    def run(self):
        while self._run_flag:
            try:
                # 라즈베리파이 접속 대기
                client, addr = self.server_socket.accept()
                self.client_socket = client
                self.client_socket.settimeout(1.0)
                self.client_connected_signal.emit(f"{addr[0]}:{addr[1]}")

                # 접속 유지 및 메시지 수신 루프
                while self._run_flag and self.client_socket:
                    try:
                        data = self.client_socket.recv(1024)
                        if not data:
                            break # 통신 끊어짐
                        msg = data.decode('utf-8').strip()
                        self.message_received_signal.emit(msg)
                    except socket.timeout:
                        continue
                    except Exception:
                        break

                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None
                self.client_disconnected_signal.emit()

            except socket.timeout:
                continue
            except Exception as e:
                break

        self.server_socket.close()

    def send_commands(self, command_list):
        """라즈베리파이로 JSON 형태의 명령 배열을 전송합니다."""
        if self.client_socket:
            try:
                payload = json.dumps({"commands": command_list}) + "\n"
                self.client_socket.sendall(payload.encode('utf-8'))
                return True
            except Exception as e:
                return False
        return False

    def stop(self):
        self._run_flag = False
        self.wait()
# ----------------------------------------

class VideoThread(QThread):
    change_pixmap_signal = Signal(QImage)
    log_signal = Signal(str)
    map_update_signal = Signal(object, int, int) 

    def __init__(self, weight_path, video_path):
        super().__init__()
        self.weight_path = weight_path
        self.video_path = video_path
        self._run_flag = True
        
        self.is_scanning = False
        self.scan_start_time = 0
        self.history_lines = []
        self.locked_lines = None
        self.current_path = [] 
        
    def start_scan(self):
        self.is_scanning = True
        self.locked_lines = None
        self.current_path = []
        self.history_lines = []
        self.scan_start_time = time.time()
        self.log_signal.emit("[시스템] 격자 스캔을 시작합니다... (3초 대기)")

    def set_path(self, path):
        self.current_path = path

    def run(self):
        vision = VisionManager(self.weight_path)
        cap = cv2.VideoCapture(self.video_path)

        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            annotated, paper_mask, edges, current_lines, object_centers = vision.process_frame(frame, self.locked_lines)

            if self.is_scanning:
                if time.time() - self.scan_start_time <= 3.0:
                    self.history_lines.append(current_lines)
                else:
                    self.is_scanning = False
                    if self.history_lines:
                        # --- [변경 로직: 오차를 허용하는 '그룹 빈도수' 계산] ---
                        # group_size = 2로 설정하면 ±1개의 오차는 같은 빈도수로 묶어줍니다.
                        # (예: 10개와 11개는 2로 나눈 몫이 5로 같으므로 같은 대세로 인정됨)
                        group_size = 1
                        
                        len_counts = Counter(len(l) // group_size for l in self.history_lines)
                        
                        # 1. 가장 자주 등장한 '개수 그룹'을 찾습니다.
                        most_common_group = len_counts.most_common(1)[0][0]
                        
                        # 2. 대세 그룹에 속하는 프레임(예: 10~11개가 찍힌 프레임들)만 후보로 모읍니다.
                        candidates = [l for l in self.history_lines if len(l) // group_size == most_common_group]
                        
                        # 3. 대세 후보들 중에서, 그래도 선이 가장 많이(뚜렷하게) 잡힌 프레임을 최종 채택!
                        self.locked_lines = max(candidates, key=len)
                        
                        self.log_signal.emit(f"[시스템] 격자 고정 완료! 선의 개수: {len(self.locked_lines)}개")

            if self.locked_lines is not None:
                grid_map, max_x, max_y, cell_centers = build_grid_map(self.locked_lines, object_centers)
                self.map_update_signal.emit(grid_map, max_x, max_y)

                if self.current_path and cell_centers:
                    for i in range(len(self.current_path) - 1):
                        pt1 = cell_centers.get(self.current_path[i])
                        pt2 = cell_centers.get(self.current_path[i+1])
                        if pt1 and pt2:
                            cv2.line(annotated, pt1, pt2, (255, 0, 255), 4)
                            cv2.circle(annotated, pt1, 6, (0, 0, 255), -1)
                    final_pt = cell_centers.get(self.current_path[-1])
                    if final_pt:
                        cv2.circle(annotated, final_pt, 10, (0, 0, 255), -1)

            rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_image = qt_image.scaled(700, 525, Qt.KeepAspectRatio) 
            self.change_pixmap_signal.emit(scaled_image)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Factory 관제 시스템")
        self.resize(1200, 850) 
        
        self.pathfinder = DijkstraPathfinder()
        self.current_grid_map = {}
        self.current_max_x = 0
        self.current_max_y = 0
        self.current_commands = [] # 전송 대기 중인 명령어 저장 리스트
        self.target_destination = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(700, 525)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label)

        panel_layout = QVBoxLayout()

        # 통신 상태창
        self.conn_label = QLabel("라즈베리파이 상태: 대기 중 🔴")
        self.conn_label.setStyleSheet("font-weight: bold; color: red; font-size: 14px;")
        panel_layout.addWidget(self.conn_label)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(200) 
        panel_layout.addWidget(QLabel("<b>[시스템 로그]</b>"))
        panel_layout.addWidget(self.log_box)

        self.btn_scan = QPushButton("격자 스캔 및 고정")
        self.btn_scan.setMinimumHeight(40)
        self.btn_scan.clicked.connect(self.action_scan)
        panel_layout.addWidget(self.btn_scan)

        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setStyleSheet("background-color: #f0f0f0; color: #333;")
        panel_layout.addWidget(QLabel("<b>[실시간 객체 및 맵 현황]</b>"))
        panel_layout.addWidget(self.status_box)

        dest_layout = QHBoxLayout()
        self.combo_block = QComboBox()
        self.combo_block.addItems(["A", "B", "C", "D", "E", "F", "G"])
        dest_layout.addWidget(QLabel("목적지:"))
        dest_layout.addWidget(self.combo_block)

        self.combo_side = QComboBox()
        self.combo_side.addItems(["1 (좌)", "2 (상)", "3 (우)", "4 (하)"])
        dest_layout.addWidget(QLabel("면:"))
        dest_layout.addWidget(self.combo_side)
        panel_layout.addLayout(dest_layout)

        self.btn_find_path = QPushButton("경로 탐색 및 명령 생성")
        self.btn_find_path.setMinimumHeight(40)
        self.btn_find_path.clicked.connect(self.action_find_path)
        panel_layout.addWidget(self.btn_find_path)

        # [신규 추가] 차량 전송 버튼
        self.btn_send_cmd = QPushButton("차량 출발 (명령 전송)")
        self.btn_send_cmd.setMinimumHeight(50)
        self.btn_send_cmd.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_send_cmd.clicked.connect(self.action_send_commands)
        self.btn_send_cmd.setEnabled(False) # 경로를 찾기 전까지는 비활성화
        panel_layout.addWidget(self.btn_send_cmd)

        main_layout.addLayout(panel_layout)
        main_layout.setStretch(0, 7) 
        main_layout.setStretch(1, 3) 

        # 영상 스레드 시작
        weight_path = r"C:\minwoin\miniproject\SmartFactory\pc\runs\segment\factory_fms\v3_segmentation\weights\best.pt"
        video_path = r"C:\minwoin\miniproject\SmartFactory\pc\example.mp4" 

        self.thread = VideoThread(weight_path, video_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.log_signal.connect(self.add_log)
        self.thread.map_update_signal.connect(self.update_map_data)
        self.thread.start()
        
        # [신규 추가] 서버 스레드 시작 (포트 8081)
        self.server_thread = ServerThread(port=8081)
        self.server_thread.client_connected_signal.connect(self.on_client_connected)
        self.server_thread.client_disconnected_signal.connect(self.on_client_disconnected)
        self.server_thread.message_received_signal.connect(self.on_message_received)
        self.server_thread.start()

        self.add_log("=== 관제 서버 부팅 완료 (포트: 8081) ===")
        self.add_log("라즈베리파이의 접속을 기다립니다...")

    def on_client_connected(self, ip_addr):
        self.conn_label.setText(f"📡 라즈베리파이 상태: 접속됨 🟢 ({ip_addr})")
        self.conn_label.setStyleSheet("font-weight: bold; color: green; font-size: 14px;")
        self.add_log(f"[네트워크] 라즈베리파이 연결 성공! ({ip_addr})")

    def on_client_disconnected(self):
        self.conn_label.setText("📡 라즈베리파이 상태: 대기 중 🔴")
        self.conn_label.setStyleSheet("font-weight: bold; color: red; font-size: 14px;")
        self.add_log("[네트워크] 라즈베리파이와의 연결이 끊어졌습니다.")

    def on_message_received(self, msg):
        """라즈베리파이로부터의 보고 수신 및 다음 명령 전송 제어"""
        # 1. 수신된 데이터(JSON) 해독
        try:
            data = json.loads(msg)
            status = data.get("status", "")
            marker_id = data.get("marker_id", "인식 안됨")
        except json.JSONDecodeError:
            # 예전 방식의 단순 텍스트("DONE")가 와도 에러가 나지 않도록 방어
            status = msg 
            marker_id = "인식 안됨"

        # 2. 로그 출력
        self.add_log(f"[수신] 라즈베리파이 보고 - 상태: {status}, 마커 ID: {marker_id}")
        
        # 3. 한 단계(Step)를 무사히 마쳤다는 신호라면?
        if status in ["STEP_DONE", "DONE"]:
            if self.current_commands:
                # 큐에 남은 명령이 있다면 다음 명령을 꺼내서 보냅니다.
                next_cmd = self.current_commands.pop(0)
                self.server_thread.send_commands([next_cmd])
                self.add_log(f"🚀 [전송] 다음 단계 시작: {next_cmd}")
            else:
                # 남은 명령이 없다면 모든 주행을 마친 것입니다!
                self.add_log("--- [시스템] 모든 주행 명령 수행 완료 ---")
                self.verify_destination() # 최종 도착 검증 실행

    def verify_destination(self):
        """YOLO 카메라 시야를 통해 최종 목적지에 제대로 도착했는지 크로스체크합니다."""
        self.add_log("--- [시스템] 카메라(YOLO) 도착 최종 검증을 시작합니다 ---")
        
        current_car_pos = None
        for coord, objects in self.current_grid_map.items():
            if 'car' in objects:
                current_car_pos = coord
                break
        
        if current_car_pos is None:
            self.add_log("🔴 [경고] 차량을 카메라에서 찾을 수 없습니다!")
            QMessageBox.critical(self, "검증 실패", "차량이 카메라 시야에서 사라졌습니다.")
            
        elif self.target_destination and current_car_pos == self.target_destination:
            self.add_log(f"🟢 [검증 성공] 차량이 목적지 {current_car_pos}에 정확히 위치해 있습니다.")
            QMessageBox.information(self, "주행 완료", "차량이 목적지에 완벽히 도착했습니다!\n다음 명령을 설정해주세요.")
            self.thread.set_path([]) # 화면의 보라색 경로선 지우기
            
        else:
            self.add_log(f"🔴 [경고] 경로 이탈! (목표: {self.target_destination}, 현재: {current_car_pos})")
            QMessageBox.warning(self, "경로 이탈", "차량이 목적지에 도착하지 못했습니다.\n수동 복구가 필요합니다.")

    def add_log(self, message):
        self.log_box.append(message)
        scrollbar = self.log_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))
        
    def update_map_data(self, grid_map, max_x, max_y):
        self.current_grid_map = grid_map
        self.current_max_x = max_x
        self.current_max_y = max_y
        
        status_text = ""
        obstacles = list("ABCDEFG")
        
        car_pos = None
        for coord, objects in grid_map.items():
            if 'car' in objects:
                car_pos = coord
                break
        status_text += f"🚗 Car 위치: {car_pos if car_pos else '찾는 중...'}\n"
        status_text += "-" * 30 + "\n"
        
        for target_obj in obstacles:
            for coord, objects in grid_map.items():
                if target_obj in objects:
                    bx, by = coord
                    available_sides = []
                    checks = {1: (bx-1, by), 2: (bx, by+1), 3: (bx+1, by), 4: (bx, by-1)}
                    for s_num, (nx, ny) in checks.items():
                        if 1 <= nx <= max_x and 1 <= ny <= max_y:
                            is_blocked = False
                            if (nx, ny) in grid_map:
                                if any(o in obstacles for o in grid_map[(nx, ny)]):
                                    is_blocked = True
                            if not is_blocked:
                                available_sides.append(s_num)
                    status_text += f"📦 {target_obj} 블록 : {coord}\n   └ 설정 가능 번호: {available_sides}\n"
        self.status_box.setText(status_text)

    def action_scan(self):
        self.pathfinder.current_heading = 0 
        self.btn_send_cmd.setEnabled(False) # 경로 초기화 시 출발 버튼 비활성화
        self.thread.start_scan()

    def action_find_path(self):
        if not self.thread.locked_lines:
            QMessageBox.warning(self, "경고", "먼저 [격자 스캔 및 고정]을 완료해주세요.")
            return

        car_pos = None
        for coord, objects in self.current_grid_map.items():
            if 'car' in objects:
                car_pos = coord
                break
                
        if car_pos is None:
            self.add_log("[오류] 차량(car)의 위치를 찾을 수 없습니다.")
            return

        target_block = self.combo_block.currentText()
        target_side = int(self.combo_side.currentText().split()[0]) 

        self.add_log(f"\n[명령 수신] 목적지: {target_block}블록, {target_side}번 면")
        
        path, msg = self.pathfinder.find_shortest_path(
            self.current_grid_map, self.current_max_x, self.current_max_y, 
            car_pos, target_block, target_side
        )
        
        if path:
            self.add_log(f"[성공] 경로: {path}")
            self.thread.set_path(path) 
            self.target_destination = path[-1]
            
            # 명령어 생성 후 변수에 저장하고 전송 버튼 활성화
            self.current_commands = self.pathfinder.generate_commands(path, target_side)
            self.add_log("--- 생성된 제어 명령 ---")
            for cmd in self.current_commands:
                self.add_log(f" * {cmd}")
            
            self.btn_send_cmd.setEnabled(True) 
        else:
            self.thread.set_path([]) 
            self.btn_send_cmd.setEnabled(False)
            self.add_log(f"[실패] {msg}")

    def action_send_commands(self):
        """[차량 출발] 버튼을 눌렀을 때 라즈베리파이로 '첫 번째' 명령만 전송"""
        if self.server_thread.client_socket is None:
            QMessageBox.critical(self, "통신 오류", "라즈베리파이가 연결되어 있지 않습니다!\n네트워크 상태를 확인해주세요.")
            return
            
        if self.current_commands:
            # 리스트에서 맨 첫 번째 명령을 꺼냅니다. (예: "앞으로 2칸")
            first_cmd = self.current_commands.pop(0) 
            
            # 리스트 형태로 감싸서 전송 (라즈베리파이 호환 유지)
            success = self.server_thread.send_commands([first_cmd])
            
            if success:
                self.add_log(f"🚀 [전송] 1단계 주행 시작: {first_cmd}")
                self.btn_send_cmd.setEnabled(False) # 주행 중 버튼 중복 클릭 방지
            else:
                self.add_log("❌ [전송 실패] 네트워크 문제로 전송에 실패했습니다.")

    def closeEvent(self, event):
        self.thread.stop()
        self.server_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())