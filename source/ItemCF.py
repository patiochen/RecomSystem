import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


# load the purchase data
purchase_data = pd.read_csv('dataset/purchase_ecommerce.dat', sep='\t', names=['visitorid', 'itemid', 'event'])

# create user-item interaction matrix
user_item_matrix = purchase_data.pivot_table(index='visitorid', columns='itemid', values='event', aggfunc='count',
                                             fill_value=0)

# convert to sparse matrix
sparse_user_item_matrix = csr_matrix(user_item_matrix)


# calculate item-item similarity
item_similarity = cosine_similarity(sparse_user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)


# set rec Function
def recommend_items(user_id, user_item_matrix, item_similarity_df, n_recommendations=5):
    if user_id not in user_item_matrix.index:
        print(f"User ID {user_id} not found in the dataset. Using a default user ID.")
        user_id = user_item_matrix.index[0]

    # get the items the user has interacted with
    user_items = user_item_matrix.loc[user_id]
    interacted_items = user_items[user_items > 0].index.tolist()

    # calculate scores for all items
    scores = pd.Series(dtype=float)
    for item in interacted_items:
        scores = scores.add(item_similarity_df[item], fill_value=0)

    # remove items the user repeated
    scores = scores.drop(interacted_items, errors='ignore')

    # get top n rec products
    recommended_items = scores.nlargest(n_recommendations).index.tolist()
    return user_id, recommended_items, interacted_items, scores


# test example(set customer )
user_id = 1
user_id, recommended_items, interacted_items, scores = recommend_items(user_id, user_item_matrix, item_similarity_df)
print(f'Recommended items for user {user_id}: {recommended_items}')
print(f'Items already interacted by user {user_id}: {interacted_items}')


# using map standard
def evaluate_map(user_id, recommended_items, user_item_matrix):
    if user_id not in user_item_matrix.index:
        return 0.0

    true_items = user_item_matrix.loc[user_id]
    true_labels = [1 if item in true_items[true_items > 0].index else 0 for item in recommended_items]
    predicted_scores = [scores[item] for item in recommended_items]
    return average_precision_score(true_labels, predicted_scores)


map_score = evaluate_map(user_id, recommended_items, user_item_matrix)
print(f'Mean Average Precision (MAP) for user {user_id}: {map_score:.4f}')

# plot figures
plt.figure(figsize=(12, 6))

# cf1
plt.bar(interacted_items, [1] * len(interacted_items), color='skyblue', label='User Interactions')
plt.bar(recommended_items, [scores[item] for item in recommended_items], color='orange', alpha=0.6,
        label='Recommended Items')
plt.xlabel('Item ID')
plt.ylabel('Score / Interactions')
plt.title('User Interactions and Recommendations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# cf2
plt.figure(figsize=(12, 6))

# get scores
nonzero_scores = scores[scores > 0].sort_index()
plt.plot(nonzero_scores.index, nonzero_scores.values,
         linewidth=2,
         color='blue',
         label='Item Scores')

# label rec products
for i, item in enumerate(recommended_items):
    score = scores[item]
    plt.scatter(item, score, color='red', s=100, zorder=5)

    # modify label locations
    if i % 2 == 0:
        xytext = (5, 15)  # up
    else:
        xytext = (5, -15)  # down

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
