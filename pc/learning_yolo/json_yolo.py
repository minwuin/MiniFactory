import json
import os
import glob

# 이미지와 제이슨 파일이 함께 있는 폴더 경로입니다.
data_dir = r"C:\minwoin\miniproject\SmartFactory\pc\factoryimage"

class_mapping = {
    "ground": 0,
    "car": 1,
    "A": 2,
    "B": 3,
    "C": 4,
    "D": 5,
    "E": 6,
    "F": 7,
    "G": 8,
}

# 지정된 폴더 내부의 모든 제이슨 파일을 탐색합니다.
json_files = glob.glob(os.path.join(data_dir, "*.json"))

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']
    
    # 파일 이름만 추출한 뒤 텍스트 파일의 경로를 동일한 폴더로 지정합니다.
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    txt_filepath = os.path.join(data_dir, base_name + ".txt")

    with open(txt_filepath, 'w', encoding='utf-8') as out_f:
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_mapping:
                continue
                
            class_id = class_mapping[label]
            shape_type = shape['shape_type']
            points = shape['points']

            normalized_points = []

            # 다각형인 경우
            if shape_type == "polygon":
                for pt in points:
                    x = pt[0] / image_width
                    y = pt[1] / image_height
                    normalized_points.extend([x, y])
                    
            # 사각형인 경우 (네 개의 꼭짓점으로 분할 연산)
            elif shape_type == "rectangle":
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                poly_points = [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ]
                
                for pt in poly_points:
                    x = pt[0] / image_width
                    y = pt[1] / image_height
                    normalized_points.extend([x, y])

            points_str = " ".join([f"{val:.6f}" for val in normalized_points])
            out_f.write(f"{class_id} {points_str}\n")
            
print(f"총 {len(json_files)}개의 파일 변환이 지정된 경로에 성공적으로 완료되었습니다.")