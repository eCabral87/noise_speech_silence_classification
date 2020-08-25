import colorednoise as cn
from matplotlib import mlab
from matplotlib import pylab as plt
import numpy as np
import os
import random
import librosa
import librosa.display
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


def generate_color_noise(samples, color):
    if color == 'white':
        beta = 0
    elif color == 'pink':
        beta = 1
    elif color == 'brown':
        beta = 2
    else:
        raise Exception('option not supported')
    return cn.powerlaw_psd_gaussian(beta, samples)

def generate_silence(samples):
    return np.zeros(samples)

def generate_speech(dataset_path):
    file = random.choice(os.listdir(dataset_path))
    return librosa.load(dataset_path + '/' + file)

def decode_number(x):
    if x == 1:
        y = 'white'
    elif x == 2:
        y = 'pink'
    elif x == 3:
        y = 'brown'
    elif x == 4:
        y = 'speech'
    else:
        y = 'silence'
    return y

def plot_signals(signal_t, fs, label):
    # Show the x(t), psd, and melspectogram of one example
    plot_noise_psd(np.array([i/fs for i in range(len(signal_t))]), signal_t, fs, label)
    get_melspectrogram(signal_t, fs/2, label)

def plot_noise_psd(time, x_signal, fs, label):
    plt.plot(time, x_signal, 'b', label=label)
    plt.xlabel('seconds')
    plt.ylabel('signal')
    plt.grid(True)
    plt.legend()
    plt.show()

    # optionally plot the Power Spectral Density with Matplotlib
    sw, f = mlab.psd(x_signal, NFFT=2 ** 9, Fs=fs)
    plt.loglog(f, sw, 'b', label=label)
    plt.xlabel('Hz')
    plt.ylabel('Pxx')
    plt.grid(True)
    plt.legend()
    plt.show()

def get_melspectrogram(y, fs, fmax, signal='y(t)'):
    melSpec = librosa.feature.melspectrogram(y=y, sr=fs, n_mels=128)
    melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(melSpec_dB, x_axis='time', y_axis='mel', sr=fs, fmax=fmax)
    plt.colorbar(format='%+1.0f dB')
    plt.title("MelSpectrogram for %s signal" % signal)
    plt.tight_layout()
    plt.show()

def get_spectral_slope_feature(signal_t, fs):
    melSpec = librosa.feature.melspectrogram(y=signal_t, sr=fs, n_mels=128)
    melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
    return np.mean(spectral_slope(melSpec_dB))

def spectral_slope(X):
    # compute mean
    mu_x = X.mean(axis=0, keepdims=True)

    # compute index vector
    kmu = np.arange(0, X.shape[0]) - X.shape[0] / 2

    # compute slope
    X = X - mu_x
    vssl = np.dot(kmu, X) / np.dot(kmu, kmu)
    return vssl

def get_spectral_cetroid_feature(melSpec, fs):
    return np.mean(librosa.feature.spectral_centroid(S=melSpec, sr=fs))

def get_mfcc_features(signal_t, fs):
    # Only return the first 13 coefficients (envelope spectrum)
    return librosa.feature.mfcc(y=signal_t, sr=fs)[:13,:]

def create_dataset(clases, speech_path, mfcc_n, fs_noise, M, N):
    # clases: number of classes of the dataset
    # speech_path: is the path where the speech recordings are located
    # mfcc_n: number of mfcc that are considered as features
    # fs_noise: sampling rate of the noise signals
    # (M, N): M is the length of the i-th signal and N is the number of signals belonging a class in the dataset

    features = {}
    features['slope'] = []
    features['centroid'] = []
    for i in range(mfcc_n):
        features['mfcc%d' % i] = []
    features['class'] = []
    for key in clases:
        for i in range(N):
            if key in ['white', 'pink', 'brown']:
                y_tmp = generate_color_noise(M, key)
                features['slope'].append(get_spectral_slope_feature(y_tmp, fs_noise))
                features['centroid'].append(get_spectral_cetroid_feature(librosa.feature.melspectrogram(y=y_tmp, sr=fs_noise, n_mels=128), fs_noise))
                mfcc = get_mfcc_features(y_tmp, fs_noise)
                for j in range(mfcc_n):
                    features['mfcc%d' % j].append(np.mean(mfcc[j, :]))
            elif key == 'speech':
                y_tmp, fs = generate_speech(speech_path)
                features['slope'].append(get_spectral_slope_feature(y_tmp, fs))
                features['centroid'].append(
                    get_spectral_cetroid_feature(librosa.feature.melspectrogram(y=y_tmp, sr=fs, n_mels=128),
                                                 fs))
                mfcc = get_mfcc_features(y_tmp, fs)
                for j in range(mfcc_n):
                    features['mfcc%d' % j].append(np.mean(mfcc[j, :]))
            else:
                y_tmp = generate_silence(M)
                features['slope'].append(get_spectral_slope_feature(y_tmp, fs_noise))
                features['centroid'].append(
                    get_spectral_cetroid_feature(librosa.feature.melspectrogram(y=y_tmp, sr=fs_noise, n_mels=128),
                                                 fs_noise))
                mfcc = get_mfcc_features(y_tmp, fs_noise)
                for j in range(mfcc_n):
                    features['mfcc%d' % j].append(np.mean(mfcc[j, :]))
            features['class'].append(key)
    return features

def normalizing_and_one_hot_encoding(dataset):
    # Normalizing
    mu = dataset.mean()
    sigma = dataset.std()
    dataset.iloc[:, :-1] = (dataset.iloc[:, :-1] - mu) / sigma

    # One hot encoding
    dataset.loc[dataset['class'] == 'white', 'class'] = 1
    dataset.loc[dataset['class'] == 'pink', 'class'] = 2
    dataset.loc[dataset['class'] == 'brown', 'class'] = 3
    dataset.loc[dataset['class'] == 'speech', 'class'] = 4
    dataset.loc[dataset['class'] == 'silence', 'class'] = 5

    x_dataset = dataset.iloc[:, :-1].values
    y_dataset = to_categorical(dataset.iloc[:, -1].values)
    return x_dataset, y_dataset,  mu, sigma

def training_NN(train_dataset, test_dataset, lr, momentum, epochs, batch_size):
    x_train_dataset, y_train_dataset, mu, sigma = normalizing_and_one_hot_encoding(train_dataset)
    x_test_dataset, y_test_dataset, _, _ = normalizing_and_one_hot_encoding(test_dataset)

    # Build NN model
    model = Sequential()
    model.add(Dense(3, activation='relu', input_dim=15))
    model.add(Dense(6, activation='softmax'))
    sgd = SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=True)
    model.summary()  # Print model Summary

    # Compile your model
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    history = model.fit(x_train_dataset, y_train_dataset, validation_data=(x_test_dataset, y_test_dataset),
                        epochs=epochs,
                        batch_size=batch_size)

    # print(history.history.keys())

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'test_loss'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'test_acc'], loc='upper left')
    plt.show()

    score = model.evaluate(x_test_dataset, y_test_dataset, batch_size=32)
    print('score: ', score)

    return model, mu, sigma

def create_model(lr, momentum):
    # Build NN model
    model = Sequential()
    model.add(Dense(3, activation='relu', input_dim=15))
    model.add(Dense(6, activation='softmax'))
    sgd = SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=True)
    model.summary()  # Print model Summary

    # Compile your model
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def predict_with_model(lr, momentum):
    # Generate signal
    fs = 22050
    samples = int(0.1 * fs)
    #x = generate_color_noise(samples, 'brown')
    x, fs = librosa.load('speech.wav')
    #x = generate_silence(samples)

    # Get features
    slope = get_spectral_slope_feature(x, fs)
    cent = get_spectral_cetroid_feature(librosa.feature.melspectrogram(y=x, sr=fs, n_mels=128), fs)
    mfcc = np.mean(get_mfcc_features(x, fs), axis=1)
    x_feature = np.array([[slope, cent, mfcc[0], mfcc[1], mfcc[2], mfcc[3], mfcc[4], mfcc[5], mfcc[6], mfcc[7], mfcc[8], mfcc[9],
                 mfcc[10], mfcc[11], mfcc[12]]])

    # Load normalization parameters
    with open('mean_std_normalization.pickle', 'rb') as f:
        mu_std = pickle.load(f)
    mu = mu_std[0].values
    std = mu_std[1].values
    x_feature = (x_feature - mu)/ std

    # Create a new model instance
    model = create_model(lr, momentum)

    # Restore the weights
    model.load_weights('./checkpoints/my_checkpoint').expect_partial()

    predict = model.predict(x_feature)
    print('prediction: ', decode_number(tf.math.argmax(predict, axis=1).numpy()[0]))

def mark_audio():
    x, fs = librosa.load('test.wav')
    time = np.array([i/fs for i in range(len(x))])
    # When shift is fixed, a larger chunk needs a longer time to compute, greater precision.
    # When chunk is fixed, a larger shift needs a shorter time to compute, poorer in precision.
    chunk = int(500e-3 * len(time) / time[len(time) - 1])  # Chunks of 500 ms
    shift = int(250e-3 * len(time) / time[len(time) - 1])  # Shift 250 ms
    w_buffer = buffer(x, chunk, chunk - shift + 1)  # num_chunk-shift+1 is equal to the overlap
    decision = np.zeros(w_buffer.shape[1], dtype=int)

    for i in range(0, w_buffer.shape[1]):
        x_tmp = np.array(w_buffer[:, i])

        # Get features
        slope = get_spectral_slope_feature(x_tmp, fs)
        cent = get_spectral_cetroid_feature(librosa.feature.melspectrogram(y=x_tmp, sr=fs, n_mels=128), fs)
        mfcc = np.mean(get_mfcc_features(x_tmp, fs), axis=1)
        x_feature = np.array(
            [[slope, cent, mfcc[0], mfcc[1], mfcc[2], mfcc[3], mfcc[4], mfcc[5], mfcc[6], mfcc[7], mfcc[8], mfcc[9],
              mfcc[10], mfcc[11], mfcc[12]]])

        # Load normalization parameters
        with open('mean_std_normalization.pickle', 'rb') as f:
            mu_std = pickle.load(f)
        mu = mu_std[0].values
        std = mu_std[1].values
        x_feature = (x_feature - mu) / std

        # Create a new model instance
        model = create_model(lr, momentum)

        # Restore the weights
        model.load_weights('./checkpoints/my_checkpoint').expect_partial()

        predict = model.predict(x_feature)
        decision[i] = tf.math.argmax(predict, axis=1).numpy()[0]

    decision = np.kron(decision, np.ones((shift,)))  # Make array decision equal to signal x
    if len(decision) >= len(x):
        decision = decision[:len(x)]
    else:
        x = x[:len(decision)]
        time = time[:len(decision)]

    # Plots results
    plt.plot(time, x, 'b', label='signal')
    plt.plot(time, decision, 'r', label='mask')
    plt.xlabel('seconds')
    plt.ylabel('signal')
    plt.grid(True)
    plt.legend()
    plt.show()

def buffer(X, n, p=0):

    '''
    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap

    Returns
    -------
    result : (n,m) ndarray
        Buffer array created from X
    '''

    d = n - p
    m = len(X)//d

    if m * d != len(X):
        m = m + 1

    Xn = np.zeros(d*m)
    Xn[:len(X)] = X

    Xn = np.reshape(Xn,(m,d))
    Xne = np.concatenate((Xn,np.zeros((1,d))))
    Xn = np.concatenate((Xn,Xne[1:,0:p]), axis = 1)

    return np.transpose(Xn[:-1])


if __name__ == "__main__":
    # Audio settings
    fs_noise = 44.1e3 # (Hz)
    duration_noise = 0.5  # (seconds)
    samples = int(duration_noise*fs_noise)  # number of samples of noise
    speech_path = './recordings'

    # NN settings
    clases = ['white', 'pink', 'brown', 'speech', 'silence']
    mfcc_n = 13
    N = 1000
    train = 0.8
    test = 0.2
    lr = 0.01
    momentum = 0.9
    epochs = 100
    batch_size = 32

    # Create dataset
    ds_features = create_dataset(clases, speech_path, mfcc_n, fs_noise, samples, N)
    df = pd.DataFrame.from_dict(ds_features)
    df.to_csv('speech-noise-silence_dataset.csv')

    # Split dataset into train and test sets
    df = pd.read_csv('speech-noise-silence_dataset.csv', index_col=0)
    train_dataset = df.groupby("class", group_keys=False).apply(pd.DataFrame.sample, frac=train)
    test_dataset = df[~df.index.isin(train_dataset.index)]

    # Training NN
    model, mu, sigma = training_NN(train_dataset, test_dataset, lr, momentum, epochs, batch_size)
    model.save_weights('./checkpoints/my_checkpoint')
    with open('mean_std_normalization.pickle', 'wb') as f:
        pickle.dump([mu, sigma], f)

    # Predict new signals
    predict_with_model(lr, momentum)

    # Mark audio based on the machine learning model
    mark_audio()




























