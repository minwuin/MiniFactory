import cv2
import numpy as np

def nothing(x):
    pass

def main():
    # 경로를 회원님의 환경에 맞게 확인해 주십시오.
    video_path = r"C:\minwoin\miniproject\SmartFactory\pc\example.mp4" 
    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow("HSV Tuner")
    
    # 검은색은 보통 명도(V)가 낮으므로 초기값을 V_MAX 80으로 세팅해두었습니다.
    cv2.createTrackbar("H_MIN", "HSV Tuner", 0, 179, nothing)
    cv2.createTrackbar("S_MIN", "HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("V_MIN", "HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("H_MAX", "HSV Tuner", 179, 179, nothing)
    cv2.createTrackbar("S_MAX", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V_MAX", "HSV Tuner", 80, 255, nothing)

    paused = False
    frame_cache = None

    print("\n=== HSV 색상 추출기 실행 ===")
    print("단축키 'p'로 화면을 멈추고 슬라이더를 조절하여 검은색 테이프만 하얗게 보이도록 맞추세요.")
    print("종료하려면 'q'를 누르세요. (종료 시 최종 값이 터미널에 출력됩니다)\n")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (width // 2, height // 2))
            frame_cache = frame.copy()

        hsv = cv2.cvtColor(frame_cache, cv2.COLOR_BGR2HSV)

        # 트랙바에서 현재 설정된 HSV 값 읽어오기
        h_min = cv2.getTrackbarPos("H_MIN", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S_MIN", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V_MIN", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H_MAX", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S_MAX", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V_MAX", "HSV Tuner")

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # HSV 마스크 생성
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        result = cv2.bitwise_and(frame_cache, frame_cache, mask=mask)

        display_frame = frame_cache.copy()
        if paused:
            cv2.putText(display_frame, "PAUSED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Original", display_frame)
        cv2.imshow("Mask (White = Tape)", mask)
        cv2.imshow("Result", result)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print(f"\n[찾은 HSV 값]")
            print(f"lower_bound = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_bound = np.array([{h_max}, {s_max}, {v_max}])")
            break
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()