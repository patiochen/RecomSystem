import pandas as pd

# Step 1: Data Preprocessing
# Load the data
data_view = pd.read_csv('dataset/view_ecommerce.dat', sep='\t', header=None, names=['visitorid', 'itemid', 'event'])
data_cart = pd.read_csv('dataset/add_to_cart_ecommerce.dat', sep='\t', header=None, names=['visitorid', 'itemid', 'event'])
data_purchase = pd.read_csv('dataset/purchase_ecommerce.dat', sep='\t', header=None, names=['visitorid', 'itemid', 'event'])

# Assign rewards based on event type
data_view['reward'] = 0
data_cart['reward'] = 0.5
data_purchase['reward'] = 1.0

# Combine all data
data = pd.concat([data_view, data_cart, data_purchase])

# Add a timestamp to simulate order
data['timestamp'] = data.groupby('visitorid').cumcount() + 1

# Define state history window size
window_size = 5

# Create state-action-reward tuples
state_action_rewards = []
for visitor, group in data.groupby('visitorid'):
    group = group.sort_values(by='timestamp')
    item_ids = list(group['itemid'])
    rewards = list(group['reward'])

    for i in range(1, len(item_ids)):
        state = item_ids[max(0, i - window_size):i]
        state = [0] * (window_size - len(state)) + state  # Pad with 0 if less than window_size
        action = item_ids[i]
        reward = rewards[i]
        state_action_rewards.append((state, action, reward))

# Save state-action-reward tuples to a text file
with open('dataset/state_action_rewards.txt', 'w') as f:
    for state, action, reward in state_action_rewards:
        f.write(f"State: {state}, Action: {action}, Reward: {reward}\n")