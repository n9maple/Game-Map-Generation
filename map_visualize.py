import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def visualize_and_save_maps(input_folder, output_folder, image_mapping):
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 獲取所有 .map 檔案
    map_files = [f for f in os.listdir(input_folder) if f.endswith('.map')]
    
    for map_file in map_files:
        map_path = os.path.join(input_folder, map_file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(map_file)[0]}.png")
        
        # 讀取 .map 檔案
        with open(map_path, 'r') as f:
            map_data = [list(line.strip()) for line in f.readlines() if line.strip()]
        
        # 將字元轉換為整數
        map_data = [[int(cell) for cell in row] for row in map_data]
        height, width = len(map_data), len(map_data[0])
        
        # 建立視覺化畫布
        fig, ax = plt.subplots(figsize=(width, height))
        ax.axis('off')
        
        # 繪製每個小格子
        for i in range(height):
            for j in range(width):
                cell_value = map_data[i][j]
                if cell_value in image_mapping:
                    img = mpimg.imread(image_mapping[cell_value])
                    ax.imshow(img, extent=[j, j+1, height-i-1, height-i], zorder=0)
        
        # 設定座標範圍，確保圖片正確顯示
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        
        # 儲存圖片
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved {output_path}")


# 圖片對應關係
image_mapping = {
    0: 'data/mountain.png',
    1: 'data/river.png',
    2: 'data/grass.png',
    3: 'data/rock.png',
    4: 'data/riverstone.png',
    5: 'data/711.jpg',
    6: 'data/house.png'
}

# 執行函式
visualize_and_save_maps('output_map', 'output', image_mapping)