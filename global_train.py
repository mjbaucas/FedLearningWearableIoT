# The code for the federated learning is from - https://github.com/dilbwagsingh/HAR-using-Federated-Learning/blob/main/notebook.ipynb
# The code of the neural network is from - https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

from numpy import mean, std, dstack, array, average, argmax
from pandas import read_csv
from matplotlib import pyplot as plt 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPool1D
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns

# plot confusion matrix
def plot(y_test, y_pred):
    confusionMatrix = confusion_matrix(y_test, y_pred)
    sns.set(font_scale=1.5)
    labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LYING"]
    plt.figure(figsize=(16,7))
    sns.heatmap(confusionMatrix, cmap = "Blues", annot = True, fmt = ".0f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class', fontsize = 20)
    plt.ylabel('Original Class', fontsize = 20)
    plt.grid()
    plt.tick_params(labelsize = 15)
    plt.xticks(rotation = 45)
    plt.show()

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a list of files and return as a 3d numpy array
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'training/HARDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'training/HARDataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

def load_models(num_models):
    models = []
    for i in range(0, num_models):
        models.append(load_model(f'./training/trained_model{i+1}.h5'))
    return models

def aggregate_weights(weights, models):
    average_model_weights = []
    n_models = len(models)
    n_layers = len(models[0].get_weights())
    for layer in range(n_layers):
        layer_weights = array([model.get_weights()[layer] for model in models])
        average_layer_weights = average(layer_weights, axis=0, weights=weights)
        average_model_weights.append(average_layer_weights)
    return average_model_weights

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, model_count):
    verbose, epochs, batch_size = 0, 10, 8
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    weights = [0.9050, 0.9114, 0.9013, 0.9125, 0.9169, 0.9104, 0.9125,0.9013, 0.8941, 0.8914] # change for each time ran with accuracies from generated models
    # weights = [max(history_1.history['accuracy']), max(history_2.history['accuracy']), max(history_3.history['accuracy'])]
    x = max(weights)
    idx = weights.index(x)
    weights[idx] = 1
    for i in range(len(weights)):
        if(weights[i] != 1):
            #weights[i] = 0.01       
            weights[i] = 0.02/(len(weights)-1)
    
    # Aggregate weights
    models = load_models(model_count)
    average_weights = aggregate_weights(weights, models)
    model.set_weights(average_weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    # model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # save weights
    model.save("global_model.h5")
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    predY = model.predict(testX)
    predY = argmax(predY, axis=1)
    plot(testy, predY)
    return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=1, model_count=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy, model_count)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

# run the experiment
run_experiment()