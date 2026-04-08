import cv2
import numpy as np
import time
from collections import Counter

# 분리된 폴더에서 클래스 호출
from vision.vision_manager import VisionManager
from navigation.pathfinder import DijkstraPathfinder

def build_grid_map(locked_lines, object_centers):
    """
    테두리가 없는 격자판에서 직선의 부등식을 활용해 객체의 좌표를 도출하고,
    동시에 화면에 경로를 그릴 수 있도록 각 셀(Cell)의 픽셀 중심점을 계산합니다.
    """
    if locked_lines is None:
        return {}, 0, 0, {}

    v_lines = []
    h_lines = []
    
    for rho, theta in locked_lines:
        angle_margin = np.pi / 8
        is_vertical = (theta < angle_margin) or (theta > np.pi - angle_margin)
        is_horizontal = abs(theta - np.pi/2) < angle_margin
        
        if is_vertical:
            v_lines.append((rho, theta))
        elif is_horizontal:
            h_lines.append((rho, theta))
            
    def get_x_intercept(line, y=240):
        rho, theta = line
        if np.cos(theta) == 0: return rho
        return (rho - y * np.sin(theta)) / np.cos(theta)
        
    def get_y_intercept(line, x=320):
        rho, theta = line
        if np.sin(theta) == 0: return rho
        return (rho - x * np.cos(theta)) / np.sin(theta)
        
    # 수직선은 화면 왼쪽에서 오른쪽으로 정렬
    v_lines.sort(key=lambda l: get_x_intercept(l, 240))
    # 수평선은 화면 아래에서 위쪽으로 정렬 (y값이 작아질수록 위쪽)
    h_lines.sort(key=lambda l: get_y_intercept(l, 320), reverse=True) 
    
    # --- [추가됨] 경로 그리기를 위한 격자 셀(Cell) 중심 픽셀 계산 ---
    v_xs = [get_x_intercept(l, 240) for l in v_lines]
    h_ys = [get_y_intercept(l, 320) for l in h_lines]

    avg_w = np.mean(np.diff(v_xs)) if len(v_xs) > 1 else 100
    avg_h = np.mean(np.abs(np.diff(h_ys))) if len(h_ys) > 1 else 100

    max_x = len(v_lines) + 1
    max_y = len(h_lines) + 1

    cell_centers = {}
    for r in range(1, max_y + 1):
        for c in range(1, max_x + 1):
            # X 좌표 중심점 계산
            if c == 1: 
                cx = v_xs[0] - avg_w / 2 if v_xs else 320
            elif c == max_x: 
                cx = v_xs[-1] + avg_w / 2 if v_xs else 320
            else: 
                cx = (v_xs[c-2] + v_xs[c-1]) / 2
                
            # Y 좌표 중심점 계산
            if r == 1: 
                cy = h_ys[0] + avg_h / 2 if h_ys else 240
            elif r == max_y: 
                cy = h_ys[-1] - avg_h / 2 if h_ys else 240
            else: 
                cy = (h_ys[r-2] + h_ys[r-1]) / 2

            cell_centers[(c, r)] = (int(cx), int(cy))
    # -----------------------------------------------------------------

    grid_map = {}
    
    for obj in object_centers:
        cls_name = obj.get('class_name', 'unknown')
        cx, cy = obj.get('center', obj.get('bottom_center', (0, 0)))
        
        col_idx = 1
        for v_line in v_lines:
            line_x = get_x_intercept(v_line, cy)
            if cx > line_x: col_idx += 1
            else: break
                
        row_idx = 1
        for h_line in h_lines:
            line_y = get_y_intercept(h_line, cx)
            if cy < line_y: row_idx += 1
            else: break
                
        coord = (col_idx, row_idx)
        if coord not in grid_map:
            grid_map[coord] = []
        grid_map[coord].append(cls_name)
        
    return grid_map, max_x, max_y, cell_centers


def main():
    weight_path = r"C:\minwoin\miniproject\SmartFactory\pc\runs\segment\factory_fms\v3_segmentation\weights\best.pt"
    video_path = r"C:\minwoin\miniproject\SmartFactory\pc\example.mp4" 

    vision = VisionManager(weight_path)
    pathfinder = DijkstraPathfinder()
    cap = cv2.VideoCapture(video_path)

    is_scanning = False
    is_locked = False
    scan_start_time = 0
    history_lines = []
    locked_lines = None
    
    current_grid_map = {}
    current_max_x, current_max_y = 0, 0
    current_cell_centers = {}
    current_path = [] # 화면에 그릴 경로 데이터 저장

    gui_logs = []
    def add_log(msg):
        gui_logs.append(msg)
        if len(gui_logs) > 12:
            gui_logs.pop(0)

    add_log("=== System Ready ===")
    add_log("[S] Scan Grid  /  [F] Find Path  /  [Q] Quit")

    print("\n=== 시스템 부팅 완료 ===", flush=True)
    print("[S] 키: 3초 격자 스캔 및 맵 고정", flush=True)
    print("[F] 키: 길 찾기 실행 및 동적 목적지 할당", flush=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, paper_mask, edges, current_lines, object_centers = vision.process_frame(frame, locked_lines)

        if is_scanning:
            if time.time() - scan_start_time <= 3.0:
                history_lines.append(current_lines)
            else:
                is_scanning = False
                is_locked = True
                if history_lines:
                    len_counts = Counter(len(l) for l in history_lines)
                    most_common_len = len_counts.most_common(1)[0][0]
                    for l in history_lines:
                        if len(l) == most_common_len:
                            locked_lines = l
                            break
                    add_log(f"Map Locked! Lines: {len(locked_lines)}")
                    print(f"\n[시스템] 격자 고정 완료! 선의 개수: {len(locked_lines)}개", flush=True)

        if is_locked and locked_lines is not None:
            # 4개의 리턴값을 받도록 수정
            current_grid_map, current_max_x, current_max_y, current_cell_centers = build_grid_map(locked_lines, object_centers)

        # --- [추가됨] 경로 시각화 (격자 중앙을 잇는 선 긋기) ---
        if current_path and current_cell_centers:
            # 점과 점을 보라색 선으로 연결
            for i in range(len(current_path) - 1):
                pt1 = current_cell_centers.get(current_path[i])
                pt2 = current_cell_centers.get(current_path[i+1])
                if pt1 and pt2:
                    cv2.line(annotated, pt1, pt2, (255, 0, 255), 4) # 굵기 4의 보라색 선
                    cv2.circle(annotated, pt1, 6, (0, 0, 255), -1)  # 각 노드에 빨간 점 표시
            
            # 최종 목적지 도착 칸에는 더 큰 점 표시
            final_pt = current_cell_centers.get(current_path[-1])
            if final_pt:
                cv2.circle(annotated, final_pt, 10, (0, 0, 255), -1)
        # -------------------------------------------------------------

        # 상태창 렌더링
        status_panel = np.zeros((450, 600, 3), dtype=np.uint8)
        status_color = (0, 0, 255) 
        status_text = "Status: WAITING (Press 's')"
        
        if is_scanning:
            status_color = (0, 255, 255) 
            remain_time = 3.0 - (time.time() - scan_start_time)
            status_text = f"Status: SCANNING... ({remain_time:.1f}s)"
        elif is_locked:
            status_color = (0, 255, 0) 
            status_text = "Status: LOCKED (Ready for commands)"
            
        cv2.putText(status_panel, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.line(status_panel, (20, 60), (580, 60), (255, 255, 255), 1)

        y_offset = 90
        for log in gui_logs:
            cv2.putText(status_panel, log, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_offset += 25

        cv2.imshow("Status & Control Panel", status_panel)
        cv2.imshow("Final Vision", annotated)

        key = cv2.waitKey(30) & 0xFF
        
        if key in [ord('q'), ord('Q')]:
            break
        elif key in [ord('s'), ord('S')]:
            is_scanning = True
            is_locked = False
            locked_lines = None
            current_path = [] # 새로 스캔하면 기존 경로 초기화
            
            # --- [추가됨] 맵 초기화 시 차량의 방향 데이터도 0(+X 방향)으로 초기화 ---
            pathfinder.current_heading = 0 
            
            scan_start_time = time.time()
            history_lines = []
            
            add_log("Scan Started... Please wait 3 seconds.")
            print("\n[시스템] 격자 스캔을 시작합니다... (차량 방향 초기화 됨)", flush=True)
            
        elif key in [ord('f'), ord('F')]:
            if not is_locked:
                add_log("Error: Press 's' to scan the grid first!")
                print("\n[경고] 먼저 's' 키를 눌러 지도를 고정해주십시오.", flush=True)
            else:
                # --- [추가됨] Car 위치 자동 검색 및 목적지 동적 입력 ---
                car_pos = None
                for coord, objects in current_grid_map.items():
                    if 'car' in objects:
                        car_pos = coord
                        break
                
                if car_pos is None:
                    add_log("Error: 'car' not found on map!")
                    print("\n[오류] 현재 화면에서 'car'를 찾을 수 없습니다. 카메라 앵글을 확인해주세요.", flush=True)
                else:
                    print(f"\n[안내] 자동차(car)의 현재 위치를 확인했습니다: {car_pos}", flush=True)
                    print("\n[현재 맵상에 인식된 타겟 블록 목록]", flush=True)
                    obstacles = list("ABCDEFG") # 장애물로 인식할 블록들
                    
                    for coord, objects in current_grid_map.items():
                        for obj in objects:
                            if obj in obstacles:
                                bx, by = coord
                                available_sides = []
                                
                                # 각 번호가 의미하는 실제 목적지 좌표(nx, ny) 매핑
                                sides_to_check = {
                                    1: (bx - 1, by), # 좌 (X 감소)
                                    2: (bx, by + 1), # 상 (Y 증가)
                                    3: (bx + 1, by), # 우 (X 증가)
                                    4: (bx, by - 1)  # 하 (Y 감소)
                                }
                                
                                for side_num, (nx, ny) in sides_to_check.items():
                                    # 1. 목적지가 맵 경계선 안쪽인지 확인
                                    if 1 <= nx <= current_max_x and 1 <= ny <= current_max_y:
                                        # 2. 해당 자리에 다른 장애물(블록)이 없는지 확인
                                        is_blocked = False
                                        if (nx, ny) in current_grid_map:
                                            for item in current_grid_map[(nx, ny)]:
                                                if item in obstacles:
                                                    is_blocked = True
                                                    break
                                        
                                        if not is_blocked:
                                            available_sides.append(side_num)
                                
                                print(f" * {obj} : {coord} , 설정 가능한 번호: {available_sides}", flush=True)
                    print("-" * 40, flush=True)
                    # ----------------------------------------------------------------------
                    
                    # 입력을 받는 동안 영상 스트리밍은 잠시 멈춥니다(정상적인 동작).
                    target_block = input("👉 목적지 블록의 이름(A~G)을 입력하세요: ").strip().upper()
                    target_side_str = input(f"👉 '{target_block}' 블록의 몇 번 면(1~4)으로 이동할까요?: ").strip()
                    
                    try:
                        target_side = int(target_side_str)
                    except ValueError:
                        print("[경고] 면 번호가 잘못되어 기본값(2)으로 설정합니다.", flush=True)
                        target_side = 2
                        
                    add_log(f"Cmd Recv: Car({car_pos}) -> {target_block}-{target_side}")
                    
                    path, msg = pathfinder.find_shortest_path(
                        current_grid_map, current_max_x, current_max_y, car_pos, target_block, target_side
                    )
                    
                    if path:
                        current_path = path # 화면에 그리기 위해 좌표 저장
                        add_log(f"Path Found: {path}")
                        print(f"\n[성공] 도출된 좌표 경로: {path}", flush=True)
                        print("--- 차량 제어 명령 리스트 ---", flush=True)
                        for cmd in pathfinder.generate_commands(path, target_side):
                            print(f" * {cmd}", flush=True)
                    else:
                        current_path = [] # 길을 못 찾으면 화면의 선 지우기
                        add_log("Error: Path not found.")
                        print(f"\n[실패] {msg}", flush=True)
                # -------------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()