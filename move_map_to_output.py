import os
import hashlib
import shutil

def move_and_deduplicate_maps(temp_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 用於存儲地圖的哈希值，防止重複
    unique_maps = set()
    map_count = 1  # 編號從1開始
    
    for filename in os.listdir(temp_folder):
        temp_path = os.path.join(temp_folder, filename)
        
        # 確保是檔案而非資料夾
        if not os.path.isfile(temp_path):
            continue
        
        # 計算檔案的哈希值來檢查是否重複
        with open(temp_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        if file_hash not in unique_maps:
            unique_maps.add(file_hash)
            new_filename = f"map_{map_count:03d}.map"
            output_path = os.path.join(output_folder, new_filename)
            
            shutil.move(temp_path, output_path)
            map_count += 1

    print(f"已完成！共移動 {map_count - 1} 個唯一地圖到 {output_folder}")

# 使用範例
move_and_deduplicate_maps('output_temp', 'output')