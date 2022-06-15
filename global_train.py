from numpy import mean, std, dstack
from pandas import read_csv
from matplotlib import pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Flatten, Drouput
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from tensorflow import keras

def load_models(num_models):
    models = []
    for i in range(1, num_models):
        models.append(keras.model.load_model(f'./training/model+{num_models}.md5'))
    return models

def aggregate_weights(weights, models):
    average_model_weights = []
    n_models = len(models)
    n_layers = len(models[0].get_weights())
    for layer  in range(n_layers):
        layer_weights = np.array([model.get_weights()[layer] for model in models])
        average_layer_weights = np.average(layer_weights, axis=0, weights=weights)
        average_model_weights.append(average_layer_weights)
    return average_model_weights

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 5, 2
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    # Fix this.... needs to load history from somewhere
    weights = [0.91,0.91,0.91] # placeholder
    # weights = [max(history_1.history['accuracy']), max(history_2.history['accuracy']), max(history_3.history['accuracy'])]
    x = max(weights)
    idx = weights.index(x)
    weights[idx] = 1
    x = min(weights)
    idx = weights.index(x)
    weights[idx] = 0.02
    for i in range(3):
        if(weights[i] != 1 and weights[i] != 0.02):
            weights[i] = 0.03
            break
    
    # Aggregate weights
    models = load_models(4)
    average_weights = aggregate_weights(weights, models):
    model.set_weights(average_weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # save model
    model.save_weights("trained_model.md5")
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

# run the experiment
run_experiment()