import pandas as pd

# 加载数据
data_cart = pd.read_csv('dataset/add_to_cart_ecommerce.dat', sep='\t', header=None,
                        names=['visitorid', 'itemid', 'event'])
data_purchase = pd.read_csv('dataset/purchase_ecommerce.dat', sep='\t', header=None,
                            names=['visitorid', 'itemid', 'event'])

# 设置奖励值
data_cart['reward'] = 1  # 加购物车奖励为1
data_purchase['reward'] = 5  # 购买奖励为5

# 合并加购和购买数据，按用户ID排序
data_positive = pd.concat([data_cart, data_purchase])
data_positive = data_positive.sort_values(['visitorid', 'event'])

# 设置状态窗口大小
N = 5

all_records = []

# 对每个用户处理数据
for visitor, group in data_positive.groupby('visitorid'):
    # 初始状态
    current_state = [0] * N
    items = list(group['itemid'])
    rewards = list(group['reward'])

    # 获取该用户的所有商品和对应的奖励
    for i in range(len(items)):
        item = items[i]
        reward = rewards[i]

        # 记录当前状态和动作
        record = f"{current_state},{item},{reward}\n"
        all_records.append(record)

        # 更新状态（将最新的商品加入状态序列）
        current_state = current_state[1:] + [item]

# 按顺序分割数据集(70%训练，30%测试)
split_point = int(len(all_records) * 0.7)
train_records = all_records[:split_point]
test_records = all_records[split_point:]

# 保存训练数据
with open('dataset/train_data.txt', 'w') as f:
    for record in train_records:
        f.write(record)

# 保存测试数据
with open('dataset/test_data.txt', 'w') as f:
    for record in test_records:
        f.write(record)

print(f"总数据量: {len(all_records)}")
print(f"训练集数量: {len(train_records)} (70%)")
print(f"测试集数量: {len(test_records)} (30%)")

