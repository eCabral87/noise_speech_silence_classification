# noise_speech_silence_classification
Noise, speech, and silence classification based on a Neural Network model using Tensorflow

This project can be used as a reference for:
1) Create a dataset as speech-noise-silence_classification.csv.
2) Split dataset into train and test sets.
3) Train a Neural Network using Tensorflow. In checkpoints folder, it can be observed the learning model generated. Likewise, in mean_std_normalization.pickle it can be observed the mean and std vectors used for normalizing the input features vectors.
4) Predict or classify new audio signals based on the Neural Network model generated in 3.
5) Mark audio files (e.g., containing audio from the classes considered) based in the Neural Network model (e.g., for voice activity detection)

Please in noise_speech_silence_classification.py set up the configuration and comment/uncomment the neccesary lines (after __main__) to only carry out what it is needed. Additionally change the configuration variables according your needs in training_NN, create_model, predict_with_model, and mark_audio functions.
