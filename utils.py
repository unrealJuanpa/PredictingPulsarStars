import numpy as np
import pandas as pd
import tensorflow as tf


fitstep(model, X, Y, )

def csv_to_tfdataset(path):
    df = pd.read_csv(path)
    features = df.iloc[:, :-1].values
    target_class = df.iloc[:, -1].values
    label_names = df.columns.tolist()

    features = np.array(features, dtype=np.float32)
    target_class = np.array(target_class, dtype=np.uint8)

    dataset = tf.data.Dataset.from_tensor_slices((features, target_class))

    stats = {}

    for i in range(features.shape[1]):
        stats[f'feature{i}'] = {
            'mean': tf.math.reduce_mean(features[:, i]).numpy(),
            'stddev': tf.math.reduce_std(features[:, i]).numpy(),
            'min': tf.math.reduce_min(features[:, i]).numpy(),
            'max': tf.math.reduce_max(features[:, i]).numpy()
        }

    return {
        'label_names': label_names,
        'dataset': dataset,
        'stats': stats
    }




data = csv_to_tfdataset('pulsar_stars_dataset.csv')

print(data['label_names'])