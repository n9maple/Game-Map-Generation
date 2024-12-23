import matplotlib.pyplot as plt
import numpy as np

# 讀取 .map 檔案
def read_map_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 轉換每行字串為數字列表
    map_data = [list(map(int, line.strip())) for line in lines]
    return np.array(map_data)

# 視覺化地圖
def visualize_map(map_data):
    plt.figure(figsize=(10, 10))
    plt.imshow(map_data, cmap='terrain', interpolation='nearest')
    plt.colorbar(label='Tile Type')
    plt.title('Map Visualization')
    plt.show()

# 刪除最後一行和最後一列
def remove_last_row_and_col(map_data):
    return map_data[:-1, :-1]


# 保存修改後的地圖
def save_map_file(map_data, output_path):
    with open(output_path, 'w') as file:
        for row in map_data:
            file.write(''.join(map(str, row)) + '\n')

# 主程式
file_path = 'data/default.map'
output_path = 'data/new_base.map'

map_data = read_map_file(file_path)

# 修改地圖，使其變45*45
map_data = remove_last_row_and_col(map_data)
visualize_map(map_data)

# 保存地圖
save_map_file(map_data, output_path)
print(f"Modified map saved to {output_path}")