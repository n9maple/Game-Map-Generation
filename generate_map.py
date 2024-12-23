import numpy as np
import random
from tqdm import tqdm

def grass_count(map):
    count = 0
    for row in map:
        for i in row:
            if i == 2:
                count += 1
    return count

def block_random(block, change_p=0.1):
    # 建立一個新的區塊來避免直接修改輸入的 block
    new_block = block.copy()
    # 定義可以互相轉換的數字
    change_map = {2: [5, 6], 5: [2, 6], 6: [2, 5]}
    # 遍歷區塊中的每個元素
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            if new_block[i, j] in change_map:
                if np.random.random() < change_p:
                    # 隨機選擇可以轉換的數字（等機率選擇）
                    new_block[i, j] = np.random.choice(change_map[new_block[i, j]])
    return new_block
    
def mutation(map, mutation_p=0.1, block_size = 5):
    # 設定地圖大小
    rows, cols = map.shape    
    # 以5x5小塊為單位進行遍歷
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = map[i:i+block_size, j:j+block_size]  # 取得當前小塊       
            # 以mutation_p的機率決定是否進行mutation
            if np.random.rand() < mutation_p:
                map[i:i+block_size, j:j+block_size] = block_random(block)  # 使用block_random進行隨機化
    return map

def uniform_crossover(map1, map2, xover_p=0.4, block_size = 5):
    # 設定地圖大小
    rows, cols = map1.shape
    # 以5x5小塊為單位進行遍歷
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # 生成隨機數來決定是否交換
            if np.random.rand() < xover_p:
                # 進行小塊交換
                map1[i:i+block_size, j:j+block_size], map2[i:i+block_size, j:j+block_size] = \
                    map2[i:i+block_size, j:j+block_size], map1[i:i+block_size, j:j+block_size]
    return map1, map2

def fitness(map, grass, block_size=5):
    house_limit = grass*0.1
    store_limit = grass*0.02
    # 獲取地圖的行列數
    rows, cols = map.shape
    # 計算小塊數量
    num_blocks = (rows // block_size) * (cols // block_size)
    # 1. 計算房子的數量及懲罰
    house_count = np.sum(map == 6)  # 房子的數目
    house_penalty = 0
    if house_count > house_limit:
        house_penalty = (house_count - house_limit) * 5  # 罰分
    # 2. 計算商店的數量及懲罰
    store_count = np.sum(map == 5)  # 商店的數目
    store_penalty = 0
    if store_count > store_limit:
        store_penalty = (store_count - store_limit) * 5  # 罰分
    # 3. 商店之間的距離（最小距離越大越好）
    store_positions = np.argwhere(map == 5)  # 取得商店位置的索引
    store_distances = []
    for i in range(len(store_positions)):
        for j in range(i+1, len(store_positions)):
            dist = np.linalg.norm(store_positions[i] - store_positions[j])  # 計算兩商店之間的歐式距離
            store_distances.append(dist)
    store_distance_score = np.mean(store_distances) if store_distances else 0  # 平均商店之間的距離
    # 4. 每棟房子離最近商店的距離（越小越好）
    house_positions = np.argwhere(map == 6)  # 取得房子位置的索引
    house_distances = []
    for house in house_positions:
        nearest_store_dist = np.min([np.linalg.norm(house - store) for store in store_positions])  # 計算最近商店的距離
        house_distances.append(nearest_store_dist)
    house_distance_score = -np.mean(house_distances)  # 這裡距離越小越好，所以取負數
    # 最終fitness分數的計算
    fitness = 0
    fitness += min(house_count, house_limit) - house_penalty  # 房子的數目加上懲罰
    fitness += min(store_count, store_limit) - store_penalty  # 商店的數目加上懲罰
    fitness += store_distance_score * 1  # 商店之間的距離加權（假設權重為2）
    fitness += house_distance_score * 1  # 房子與商店的距離加權（假設權重為3）
    return fitness

def compute_fitness_for_list(map_list, grass):
    fitness_list = []
    for map in map_list:
        fitness_list.append(fitness(map, grass))
    return fitness_list

def roulette_wheel_selection(map_list, fitness_list):
    # 計算所有地圖的適應度總和
    total_fitness = np.sum(fitness_list)
    # 計算每張地圖的選擇機率
    probabilities = fitness_list / total_fitness 
    # 根據選擇機率生成累積機率
    cumulative_probabilities = np.cumsum(probabilities)
    # 隨機選擇一張地圖
    rand = np.random.random()
    selected_index = np.searchsorted(cumulative_probabilities, rand)
    # 回傳選擇的地圖
    selected_map = map_list[selected_index]
    return selected_map

def tournament_selection(map_list, fitness_list, tournament_size=30):
    # 隨機選擇 tournament_size 個索引
    indices = random.sample(range(len(fitness_list)), tournament_size)
    # 根據索引選擇適應度和地圖
    selected_fitness = [fitness_list[i] for i in indices]
    selected_maps = [map_list[i] for i in indices]
    # 找到適應度最高的索引
    best_index = selected_fitness.index(max(selected_fitness))
    # 返回對應的地圖
    return selected_maps[best_index]

# 讀取 .map 檔案
def read_map_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 轉換每行字串為數字列表
    map_data = [list(map(int, line.strip())) for line in lines]
    return np.array(map_data)

# 保存修改後的地圖
def save_map_file(map_data, output_path):
    with open(output_path, 'w') as file:
        for row in map_data:
            file.write(''.join(map(str, row)) + '\n')

def select_top_maps(map_list, fitness_list, top_k=10):
    # 取得前 top_k 個最大 fitness 的索引
    top_indices = np.argsort(fitness_list)[-top_k:][::-1]  # 由大到小排序
    
    # 根據索引挑選對應的 map
    top_maps = [map_list[i] for i in top_indices]
    top_fitness = [fitness_list[i] for i in top_indices]
    
    return top_fitness, top_maps

def initialization(base_map, population, block_size = 5):
    rows, cols = base_map.shape
    map_list = []
    for i in range(population):
        map = base_map.copy()
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                map[i:i+block_size, j:j+block_size] = block_random(map[i:i+block_size, j:j+block_size])
        map_list.append(map)
    return map_list

def main():
    generation = 20
    population = 50
    base_map = read_map_file('data/new_base.map')
    grass = grass_count(base_map)
    map_list = initialization(base_map, population)
    fitness_list = compute_fitness_for_list(map_list, grass)
    for i in tqdm(range(generation)):
        new_map_list = []
        while len(new_map_list) < population:
            parent_1 = roulette_wheel_selection(map_list, fitness_list)
            parent_2 = roulette_wheel_selection(map_list, fitness_list)
            child1, child2 = uniform_crossover(parent_1, parent_2)
            new_map_list.append(child1)
            new_map_list.append(child2)
        for ind, map in enumerate(new_map_list):
            new_map_list[ind] = mutation(map)
        map_list = new_map_list
        fitness_list = compute_fitness_for_list(map_list, grass)
    
    # save map
    top_k = 50
    top_fitness, top_maps = select_top_maps(map_list, fitness_list, top_k)
    for i in range(top_k):
        print(f'map{i+1} has fitness value {top_fitness[i]}')
        save_map_file(top_maps[i], f'output_temp/new3_{i+1}.map')

if __name__ == '__main__':
    main()