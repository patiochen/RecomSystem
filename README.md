# DQN-Based Recommendation System

## Introduction
This is the final project for CSCE642. The project implements a recommendation system using a basic Deep Q-Network (DQN) model. The codebase provides a complete framework for data preprocessing, training, and test of the recommendation system on real e-commerce data. This README file will guide you through setting up the environment, running the code, and understanding the results.

## Prerequisites

To use the code, you need the following installed:

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- Anaconda
- pandas

## Dataset

All dataset files are saved in the dataset folder. We utilized a real-world dataset collected by RetailRocket from an e-commerce website: [source](https://github.com/caserec/Datasets-for-Recommender-Systems/blob/master/Processed%20Datasets/RetailrocketEcommerce). The dataset comprises 92,490 interactions between 3,431 users and 8,885 items, with each user having viewed at least 10 items. These interactions include viewing items, adding items to the cart, and purchasing items.

The original dataset is organized into three separate data files based on interaction types:

- `add_to_cart_ecommerce.dat`: Dataset of items added to the cart by customers (used in our project)
- `purchase_ecommerce.dat`: Records of items purchased by customers (used in our project)

The other two datasets can be ignored for the purposes of this project.

## Function Descriptions

All project-related functions are located in the `source` folder:

- **ItemCF.py**: Implements the ItemCF-based recommendation system. It can be run independently and outputs a statistical graph of the items recommended by the ItemCF algorithm.

- **DQN-based RS**: Consists of four function files:
  - `data_preprocessing.py`: Processes the dataset files and merges them into a format suitable for the recommendation system.
  - `env.py`: Sets up the environment for the recommendation system.
  - `dqn.py`: Implements the DQN algorithm.
  - `test.py`: Tests the recommendation system using the model trained in `dqn.py` and the test dataset.

To run the DQN-based recommendation system, first run `data_preprocessing.py` to generate `train_data` and `test_data`. Then, run `dqn.py` to train the model, which will save as `recommender_model.pth`. Finally, run `test.py`. Both `dqn.py` and `test.py` will generate two statistical graphs of the recommendation results, saved in the `source` folder.

## How to Run the Code

### 1. Data Preprocessing

Before training, make sure to preprocess the data by running the following command:

```sh
python source/data_preprocessing.py
```

This script will clean the data and convert it into a suitable format for the DQN model.

### 2. Training the Model

To train the DQN model, run the following command:

```sh
python dqn.py
```

This will start the training process, which uses the training data to train a Deep Q-Network for the recommendation system. The model parameters will be saved to `recommender_model.pth` upon completion.

### 3. Testing the Model

To evaluate the trained model on the test dataset, run the following command:

```sh
python test.py
```

This script will visualize the recommendation results for a single episode, showing the recommended items and corresponding rewards.

### 4. Visualizing Results

The code files `dqn.py` and `test.py` will generate two statistical graphs `.png` files of the recommendation results, saved in the `source` folder.

- `training_metrics.png`: Shows the loss, average reward, and Q-value changes during training.
- `episode_timeline.png`: Shows the recommended items and rewards for a single episode.


## Contribution

The technical development of this project was carried out collaboratively by the team, as detailed in the report's contribution statement.

## Special Note

Please note that the recorded video in Youtube does not cover the complete results of the project. For the latest results, please refer to the project report (Section 7).

