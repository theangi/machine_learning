import os
import shutil
import requests

import numpy as np
import tensorflow as tf

MODEL_TEMP_FILE = '/tmp/iris_model'

IRIS_TRAINING = 'iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = 'iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

CLASS_IRIS_SETOSA = b'0'
CLASS_IRIS_VERSICOLOR = b'1'
CLASS_IRIS_VIRGINICA = b'2'


def download_sets_if_necessary():
    if not os.path.exists(IRIS_TRAINING):
        raw_data = requests.get(IRIS_TRAINING_URL)
        with open(IRIS_TRAINING, 'wb') as f:
            f.write(raw_data.content)
            print('Training set download')

    if not os.path.exists(IRIS_TEST):
        raw_data = requests.get(IRIS_TEST_URL)
        with open(IRIS_TEST, 'wb') as f:
            f.write(raw_data.content)
            print('Test set download')


def get_human_result(numeric_class):
    if np.asscalar(numeric_class) == CLASS_IRIS_SETOSA:
        return 'Iris Setosa'
    elif np.asscalar(numeric_class) == CLASS_IRIS_VERSICOLOR:
        return 'Iris Versicolor'
    elif np.asscalar(numeric_class) == CLASS_IRIS_VIRGINICA:
        return 'Iris Virginica'
    return 'Unknow class!'


def plot_data():
    with open(IRIS_TRAINING, 'r') as f:
        print('Header: {}'.format(f.readline().replace('\n', '')))
        print('Data: {}'.format(f.readline().replace('\n', '')))
        print('Data: {}'.format(f.readline().replace('\n', '')))
        print('Data: ...')


def remove_model_file():
    try:
        shutil.rmtree(MODEL_TEMP_FILE)
    except FileNotFoundError as e:
        print('Cannot delete {}. {}'.format(MODEL_TEMP_FILE, e))


def prepare_sets():
    download_sets_if_necessary()

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    return training_set, test_set


def get_accuracy_for_dnn(training_set, test_set, hidden_units, steps):
    number_of_features = 4

    feature_columns = [tf.feature_column.numeric_column('x', shape=[number_of_features])]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        n_classes=3,
        model_dir=MODEL_TEMP_FILE)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    classifier.train(input_fn=train_input_fn, steps=steps)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    accuracy_score = classifier.evaluate(input_fn=test_input_fn)['accuracy']
    return accuracy_score

    # new_samples = np.array(
    #   [[6.4, 3.2, 4.5, 1.5],
    #    [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    #
    # predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    #   x={'x': new_samples},
    #   num_epochs=1,
    #   shuffle=False)
    #
    # predictions = list(classifier.predict(input_fn=predict_input_fn))
    # for p in predictions:
    #     print('Class: {} (since probabilities were {})'.format(get_human_result(p['classes']), p['probabilities']))


if __name__ == '__main__':
    training, test = prepare_sets()

    results = {}
    try:
        for x in range(5, 30):
            for y in range(5, 30):
                for z in range(5, 30):
                    print('Processing {}, {}, {}'.format(x, y, z))
                    remove_model_file()
                    result = get_accuracy_for_dnn(training, test, hidden_units=[x, y, z], steps=1000)
                    if result not in results:
                        results[result] = (x, y, z)
        print(results)
    except KeyboardInterrupt as e:
        # For ctrl + c when you are done :)
        print(results)

