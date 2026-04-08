import socket, cv2, numpy as np, threading, time
import cv2.aruco as aruco

# --- [설정 및 초기화] ---
# ⚠️ 라즈베리파이의 IP 주소를 적어주세요! (터미널에서 hostname -I 로 확인 가능)
RASP_IP = '192.168.137.1' # <--- 이 부분을 꼭 확인 후 수정하세요!

frame_f, frame_s = None, None
auto_mode = False
robot_stage = "TRACKING" 
selected_side = None

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

def receive_v(port, mode):
    global frame_f, frame_s
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('0.0.0.0', port))
    while True:
        try:
            data, _ = s.recvfrom(65535)
            img = cv2.imdecode(np.frombuffer(data[4:], dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                if mode == 'F': frame_f = img
                else: frame_s = img
        except: pass

threading.Thread(target=receive_v, args=(5002, 'F'), daemon=True).start()
threading.Thread(target=receive_v, args=(5003, 'S'), daemon=True).start()

client_motor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try: 
    print(f"📡 라즈베리파이({RASP_IP})에 연결을 시도합니다...")
    client_motor.connect((RASP_IP, 8080))
    print("🟢 연결 성공! 조작 창을 띄웁니다.")
except: 
    print("❌ 연결 실패! 라즈베리파이 IP를 확인하거나, 라즈베리파이 서버가 켜져 있는지 확인하세요.")

def send_cmd(msg):
    try:
        client_motor.sendall(msg.encode())
        res = client_motor.recv(1024)
        time.sleep(0.04)
        return res
    except: return None

# 제어 파라미터
PWR_L, PWR_R = 0.45, 0.50
MOVE_TICK = 0.05
STOP_LINE_Y = 165  
CENTER_X = 320     
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

print("\n==================================")
print(" 🎮 [조작 단축키 가이드]")
print(" - 'a' : 자율주행(자동 모드) ON/OFF")
print(" - 'w' : (수동) 강제 1.4초 전진")
print(" - '[' : 교차로 좌측 측면 스캔 모드")
print(" - ']' : 교차로 우측 측면 스캔 모드")
print(" - 'r' : 강제 정지 및 추적 모드 초기화")
print(" - 'q' : 시스템 완전히 끄기")
print("==================================\n")

while True:
    if auto_mode:
        if robot_stage == "TRACKING" and frame_f is not None:
            disp_f = cv2.resize(frame_f, (640, 480))
            gray_f = clahe.apply(cv2.cvtColor(disp_f, cv2.COLOR_BGR2GRAY))
            
            corners, ids, _ = detector.detectMarkers(gray_f)
            qr_center_x, qr_center_y = None, None
            
            if ids is not None:
                c = corners[0][0]
                qr_center_x = int(np.mean(c[:, 0]))
                qr_center_y = int(np.mean(c[:, 1]))
                cv2.circle(disp_f, (qr_center_x, qr_center_y), 5, (0, 255, 255), -1)
                aruco.drawDetectedMarkers(disp_f, corners, ids)

            thresh = cv2.adaptiveThreshold(gray_f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)
            n, _, stats, cents = cv2.connectedComponentsWithStats(thresh)
            line_cx = int(cents[np.argmax(stats[1:, 4]) + 1, 0]) if n > 1 else None

            if qr_center_y is not None:
                if qr_center_y >= STOP_LINE_Y and abs(qr_center_x - CENTER_X) < 20:
                    send_cmd("0")
                    robot_stage = "WAIT_BTN"
                    auto_mode = False
                    print(f"🎯 교차점 정지 성공! (X:{qr_center_x}, Y:{qr_center_y})")
                else:
                    err = qr_center_x - CENTER_X
                    if abs(err) > 15: 
                        send_cmd(f"{'4' if err > 0 else '3'},0.02")
                    else:
                        send_cmd(f"1,{PWR_L},{PWR_L},{PWR_R},{PWR_R},{MOVE_TICK}")
            
            elif line_cx is not None: 
                err = line_cx - CENTER_X
                if abs(err) > 20:
                    send_cmd(f"{'4' if err > 0 else '3'},0.02")
                else:
                    send_cmd(f"1,{PWR_L},{PWR_L},{PWR_R},{PWR_R},{MOVE_TICK}")

        elif robot_stage == "SIDE_SCAN" and frame_s is not None:
            disp_s = cv2.resize(frame_s, (640, 480))
            gray_s = clahe.apply(cv2.cvtColor(disp_s, cv2.COLOR_BGR2GRAY))
            corners_s, ids_s, _ = detector.detectMarkers(gray_s)
            
            s_qr_cx = int(np.mean(corners_s[0][0][:, 0])) if ids_s is not None else None
            
            if s_qr_cx is not None and abs(s_qr_cx - CENTER_X) < 30:
                print("🎯 측면 정렬 완료! 0.9초 돌파!")
                send_cmd("1,0.4,0.22,0.4,0.2, 1.0")
                auto_mode = False; robot_stage = "TRACKING"
            else:
                cmd = '3' if selected_side == "LEFT" else '4'
                send_cmd(f"{cmd},0.02")

    if frame_f is not None:
        disp_f = cv2.resize(frame_f, (640, 480))
        cv2.line(disp_f, (0, STOP_LINE_Y), (640, STOP_LINE_Y), (0, 0, 255), 2)
        cv2.line(disp_f, (CENTER_X, 0), (CENTER_X, 480), (255, 0, 0), 1)
        cv2.circle(disp_f, (CENTER_X, STOP_LINE_Y), 10, (0, 255, 0), 2)
        cv2.imshow('FRONT_CSI (Press keys here)', disp_f)
        
    if frame_s is not None: cv2.imshow('SIDE_WEBCAM', cv2.resize(frame_s, (640, 480)))

    # 키보드 입력 감지 (OpenCV 창이 선택된 상태에서 눌러야 합니다)
    key = cv2.waitKey(1)
    if key == ord('w'): send_cmd("1,0.4,0.22,0.4,0.2,1.4")
    elif key == ord('a'):
        auto_mode = not auto_mode
        if auto_mode and robot_stage == "WAIT_BTN": robot_stage = "TRACKING"
        print(f"🤖 자동 모드: {'켜짐' if auto_mode else '꺼짐'}")
    elif key == ord('r'): robot_stage = "TRACKING"; auto_mode = False; send_cmd("0"); print("🛑 강제 정지!")
    elif key == ord('['): selected_side = "LEFT"; robot_stage = "SIDE_SCAN"; auto_mode = True; print("🔍 좌측 스캔 모드 가동")
    elif key == ord(']'): selected_side = "RIGHT"; robot_stage = "SIDE_SCAN"; auto_mode = True; print("🔍 우측 스캔 모드 가동")
    elif key == ord('q'): break

cv2.destroyAllWindows()