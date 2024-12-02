import pandas as pd

# load dataset
data_cart = pd.read_csv('dataset/add_to_cart_ecommerce.dat', sep='\t', header=None,
                        names=['visitorid', 'itemid', 'event'])
data_purchase = pd.read_csv('dataset/purchase_ecommerce.dat', sep='\t', header=None,
                            names=['visitorid', 'itemid', 'event'])

# set reward
data_cart['reward'] = 1  # add-cart +1
data_purchase['reward'] = 5  # buy +5

# merge data
data_positive = pd.concat([data_cart, data_purchase])
data_positive = data_positive.sort_values(['visitorid', 'event'])

# set window N=5
N = 5

all_records = []

# duel with each data
for visitor, group in data_positive.groupby('visitorid'):
    # set initial state
    current_state = [0] * N
    items = list(group['itemid'])
    rewards = list(group['reward'])

    # get all products and reward for each visitorid
    for i in range(len(items)):
        item = items[i]
        reward = rewards[i]

        # record state, actions
        record = f"{current_state},{item},{reward}\n"
        all_records.append(record)

        # update state
        current_state = current_state[1:] + [item]

# set training set and test set 7:3
split_point = int(len(all_records) * 0.7)
train_records = all_records[:split_point]
test_records = all_records[split_point:]

# save data
with open('dataset/train_data.txt', 'w') as f:
    for record in train_records:
        f.write(record)

with open('dataset/test_data.txt', 'w') as f:
    for record in test_records:
        f.write(record)

# print(f"总数据量: {len(all_records)}")
# print(f"训练集数量: {len(train_records)} (70%)")
# print(f"测试集数量: {len(test_records)} (30%)")

