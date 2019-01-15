import dataset
import tensorflow as tf
import numpy as np
import test

training_split = 0.95

def main():
    data = dataset.Dataset("/Users/Huw/Documents/GitHub/Gambling_Predictor/data/book.csv",5)

    train_data_len = int(training_split * len(data.processed_results))
    train_data = data.processed_results[:train_data_len]
    test_data = data.processed_results[train_data_len:]


    def features_labels(datas):
        features = {}

        for d in datas:
            for key in d.keys():
                if key not in features:
                    features[key] = []

                features[key].append(d[key])

        for key in features.keys():
            features[key] = np.array(features[key])

        return features, features['result']

    train_features, train_labels = features_labels(train_data)
    test_features, test_labels = features_labels(test_data)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_features,
        y=train_labels,
        batch_size=500,
        num_epochs=None,
        shuffle=True
    )

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_features,
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )

    feature_columns = []

    for mode in ['home', 'away']:
        feature_columns = feature_columns + [
            tf.feature_column.numeric_column(key='{}-wins'.format(mode)),
            tf.feature_column.numeric_column(key='{}-draws'.format(mode)),
            tf.feature_column.numeric_column(key='{}-losses'.format(mode)),
            tf.feature_column.numeric_column(key='{}-goals'.format(mode)),
            tf.feature_column.numeric_column(key='{}-opposition-goals'.format(mode)),
            tf.feature_column.numeric_column(key='{}-shots'.format(mode)),
            tf.feature_column.numeric_column(key='{}-shots-on-target'.format(mode)),
            tf.feature_column.numeric_column(key='{}-opposition-shots'.format(mode)),
            tf.feature_column.numeric_column(key='{}-opposition-shots-on-target'.format(mode)),
        ]


    model = tf.estimator.DNNClassifier(
        model_dir='model/',
        hidden_units=[10],
        feature_columns=feature_columns,
        n_classes=3,
        label_vocabulary=['H', 'D', 'A'],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001
        ))


    for i in range(0, 50):
        model.train(input_fn=train_input_fn, steps=100)
        print(i*100)
    evaluation_result = model.evaluate(input_fn=test_input_fn)
    predictions = list(model.predict(input_fn=test_input_fn))
    test.calculate_accuracy(predictions, test_labels, test_features)

main()
