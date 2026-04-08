import heapq

class DijkstraPathfinder:
    def __init__(self):
        # [수정됨] 1:좌, 2:상, 3:우, 4:하 (Y축 위로 갈수록 증가하는 데카르트 좌표계)
        self.side_offsets = {
            1: (-1, 0), # 왼쪽 면 (X 감소)
            2: (0, 1),  # 위쪽 면 (Y 증가)
            3: (1, 0),  # 오른쪽 면 (X 증가)
            4: (0, -1)  # 아래쪽 면 (Y 감소)
        }
        # 추가됨: 차량의 현재 방향 상태를 기억 (0: 우, 1: 상, 2: 좌, 3: 하)
        self.current_heading = 0

    def find_shortest_path(self, grid_map, max_x, max_y, start_pos, target_block, target_side):
        target_pos = None
        for coords, objects in grid_map.items():
            if target_block in objects:
                target_pos = coords
                break
        
        if not target_pos:
            return None, "타겟 블록을 찾을 수 없습니다."

        offset = self.side_offsets.get(target_side, (0, 0))
        goal_pos = (target_pos[0] + offset[0], target_pos[1] + offset[1])

        obstacles = set()
        for coords, objects in grid_map.items():
            for obj in objects:
                if obj in "ABCDEFG":
                    obstacles.add(coords)

        if not (1 <= goal_pos[0] <= max_x and 1 <= goal_pos[1] <= max_y):
            return None, f"목적지({goal_pos})가 맵 경계를 벗어납니다."
        if goal_pos in obstacles:
            return None, f"목적지({goal_pos})에 다른 장애물 블록이 있어 도달할 수 없습니다."

        queue = [(0, start_pos, [])] 
        visited = set()

        while queue:
            (cost, current, path) = heapq.heappop(queue)
            if current in visited: continue
            visited.add(current)
            path = path + [current]

            if current == goal_pos:
                return path, "경로 탐색 성공"

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_node = (current[0] + dx, current[1] + dy)
                if (1 <= next_node[0] <= max_x and 1 <= next_node[1] <= max_y and 
                    next_node not in obstacles and next_node not in visited):
                    heapq.heappush(queue, (cost + 1, next_node, path))

        return None, "장애물에 막혀 갈 수 있는 경로가 없습니다."

    def generate_commands(self, path, target_side):
        """
        요구하신 형식으로 명령을 생성합니다: "앞으로 N칸", "90도 [좌/우]회전 1회"
        이동이 끝난 후 차량의 최종 방향을 클래스 변수에 저장하여 다음 이동 시 기준으로 삼습니다.
        """
        if not path or len(path) < 2:
            return ["현재 위치가 목적지입니다."]

        # 매번 초기화하지 않고, 저장되어 있던 현재 차량의 방향을 불러옵니다.
        current_heading = self.current_heading 
        forward_count = 0
        final_commands = []

        for i in range(len(path) - 1):
            curr = path[i]
            nxt = path[i+1]
            
            dx = nxt[0] - curr[0]
            dy = nxt[1] - curr[1]
            
            # 다음 이동을 위한 목표 방위각 설정 (0: 우, 1: 상, 2: 좌, 3: 하)
            target_heading = 0
            if dx == 1: target_heading = 0
            elif dy == 1: target_heading = 1
            elif dx == -1: target_heading = 2
            elif dy == -1: target_heading = 3
            
            # 현재 방향과 가야 할 방향 사이의 회전량 계산
            turn = (target_heading - current_heading) % 4
            
            if turn != 0:
                # 회전하기 전, 쌓여있던 직진 명령을 먼저 확정
                if forward_count > 0:
                    final_commands.append(f"앞으로 {forward_count}칸")
                    forward_count = 0
                
                # 회전 명령 추가
                if turn == 1:
                    final_commands.append("90도 좌회전 1회")
                elif turn == 2:
                    final_commands.append("90도 좌회전 1회")
                    final_commands.append("90도 좌회전 1회")
                elif turn == 3:
                    final_commands.append("90도 우회전 1회")
                
                # 차량의 방향 업데이트
                current_heading = target_heading
            
            # 직진 카운트 증가
            forward_count += 1

        # 마지막으로 남아있는 직진 명령 추가
        if forward_count > 0:
            final_commands.append(f"앞으로 {forward_count}칸")

        # --- 도착 후 타겟 블록을 바라보도록 최종 회전 ---
        look_heading_map = {1: 0, 2: 3, 3: 2, 4: 1}
        final_look_heading = look_heading_map.get(target_side, current_heading)
        
        final_turn = (final_look_heading - current_heading) % 4
        if final_turn == 1:
            final_commands.append("90도 좌회전 1회 (블록 조준)")
        elif final_turn == 2:
            final_commands.append("90도 좌회전 1회")
            final_commands.append("90도 좌회전 1회 (블록 조준)")
        elif final_turn == 3:
            final_commands.append("90도 우회전 1회 (블록 조준)")

        # --- [핵심] 주행과 조준이 모두 끝난 후의 최종 방향을 메모리에 저장 ---
        self.current_heading = final_look_heading

        return final_commands