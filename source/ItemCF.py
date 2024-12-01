import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

# Step 1: Data Loading and Preprocessing
# Load the purchase data
purchase_data = pd.read_csv('dataset/purchase_ecommerce.dat', sep='\t', names=['visitorid', 'itemid', 'event'])

# Create user-item interaction matrix
user_item_matrix = purchase_data.pivot_table(index='visitorid', columns='itemid', values='event', aggfunc='count',
                                             fill_value=0)

# Convert to sparse matrix
sparse_user_item_matrix = csr_matrix(user_item_matrix)

# Step 2: Item-based Collaborative Filtering
# Calculate item-item similarity
item_similarity = cosine_similarity(sparse_user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)


# Step 3: Recommendation Function
def recommend_items(user_id, user_item_matrix, item_similarity_df, n_recommendations=5):
    if user_id not in user_item_matrix.index:
        print(f"User ID {user_id} not found in the dataset. Using a default user ID.")
        user_id = user_item_matrix.index[0]

    # Get the items the user has interacted with
    user_items = user_item_matrix.loc[user_id]
    interacted_items = user_items[user_items > 0].index.tolist()

    # Calculate scores for all items
    scores = pd.Series(dtype=float)
    for item in interacted_items:
        scores = scores.add(item_similarity_df[item], fill_value=0)

    # Remove items the user has already interacted with
    scores = scores.drop(interacted_items, errors='ignore')

    # Get top n recommendations
    recommended_items = scores.nlargest(n_recommendations).index.tolist()
    return user_id, recommended_items, interacted_items, scores


# Step 4: Example Recommendation
user_id = 1
user_id, recommended_items, interacted_items, scores = recommend_items(user_id, user_item_matrix, item_similarity_df)
print(f'Recommended items for user {user_id}: {recommended_items}')
print(f'Items already interacted by user {user_id}: {interacted_items}')


# Step 5: Evaluation using Mean Average Precision (MAP)
def evaluate_map(user_id, recommended_items, user_item_matrix):
    if user_id not in user_item_matrix.index:
        return 0.0

    true_items = user_item_matrix.loc[user_id]
    true_labels = [1 if item in true_items[true_items > 0].index else 0 for item in recommended_items]
    predicted_scores = [scores[item] for item in recommended_items]
    return average_precision_score(true_labels, predicted_scores)


map_score = evaluate_map(user_id, recommended_items, user_item_matrix)
print(f'Mean Average Precision (MAP) for user {user_id}: {map_score:.4f}')

# Step 6: 分开的可视化图表
plt.figure(figsize=(12, 6))

# 第一张图：柱状图
plt.bar(interacted_items, [1] * len(interacted_items), color='skyblue', label='User Interactions')
plt.bar(recommended_items, [scores[item] for item in recommended_items], color='orange', alpha=0.6,
        label='Recommended Items')
plt.xlabel('Item ID')
plt.ylabel('Score / Interactions')
plt.title('User Interactions and Recommendations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 第二张图：平滑的折线图
plt.figure(figsize=(12, 6))

# 获取非零得分的商品和得分
nonzero_scores = scores[scores > 0].sort_index()
plt.plot(nonzero_scores.index, nonzero_scores.values,
         linewidth=2,
         color='blue',
         label='Item Scores')

# 标注推荐的商品，使用不同的标签位置
for i, item in enumerate(recommended_items):
    score = scores[item]
    plt.scatter(item, score, color='red', s=100, zorder=5)

    # 交替标签位置：一上一下
    if i % 2 == 0:
        xytext = (5, 15)  # 向上偏移
    else:
        xytext = (5, -15)  # 向下偏移

    plt.annotate(f'Item {item}',
                 (item, score),
                 xytext=xytext,
                 textcoords='offset points',
                 ha='left',
                 va='center')

plt.xlabel('Item ID')
plt.ylabel('Similarity Score')
plt.title('Item Recommendation Scores')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
