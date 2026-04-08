import cv2
import socket
import numpy as np

def receive_bottom_camera():
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5003  # 하단 카메라 전용 포트

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print(f"📡 [PC 수신기] {UDP_PORT}번 포트에서 하단 카메라 영상을 기다리는 중...")
    print("종료하려면 영상 창을 마우스로 클릭한 뒤 'q' 키를 누르세요.")

    while True:
        try:
            # 라즈베리파이에서 보낸 데이터 수신
            data, addr = sock.recvfrom(65535)
            
            # 동료분들 코드 규칙: 앞의 4바이트(길이)를 잘라내고 순수 이미지만 디코딩
            img_data = np.frombuffer(data[4:], dtype=np.uint8)
            frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # 화면이 너무 작을 수 있으므로 640x480으로 확대해서 보여줌
                frame = cv2.resize(frame, (640, 480))
                cv2.imshow("AGV Bottom Camera (Port: 5003)", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"❌ 수신 오류 발생: {e}")
            break

    cv2.destroyAllWindows()
    sock.close()

if __name__ == "__main__":
    receive_bottom_camera()