import random
import numpy as np

def poison_data(data, percent=0.1, noise_type='random'):
    data = data.copy()
    n_poison = int(len(data) * percent)

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    poison_indices = random.sample(range(len(data)), n_poison)

    for idx in poison_indices:
        if noise_type == 'random':
            # Generate one random value per feature
            random_values = np.random.uniform(
                low=data[feature_cols].min().values,
                high=data[feature_cols].max().values
            )
            data.loc[idx, feature_cols] = random_values

        # Optional: flip label (uncomment below if using label encoding: 0, 1, 2)
        # labels = [0, 1, 2]
        # current_label = data.loc[idx, 'species']
        # new_label = random.choice([l for l in labels if l != current_label])
        # data.loc[idx, 'species'] = new_label
    print("data poisoned")
    return data


