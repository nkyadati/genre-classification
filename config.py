"""Model config in json format"""

CFG = {
    "data": {
        "path": "./musimotion_testset_annotations_cut.csv",
        "dataset_path": "./musimotion_youtube_subtestset",
        "json_path": "./",
        "test_split": 0.2
    },
    "features": {
        "save": True,
        "duration": 120,
        "sampling_rate": 22050,
        "num_segments": 10,
        "feature_type": "mfcc",
        "mfcc": {
            "num_mfcc": 13,
        },
        "melspectrogram": {
            "n_mels": 128
        },
        "n_fft": 2048,
        "hop_length": 512,
    },
    "train": {
        "batch_size": 32,
        "epochs": 100,
        "optimizer": {
            "type": "adam",
            "learning_rate": 0.0001,
        },
        "loss": "binary_crossentropy",
    },
    "model": {
        "model_type": "lstm",
        "lstm_layer": {
            "num_lstm_layers": 2,
            "num_units": [32, 32]
        },
        "conv_layer": {
            "num_conv_layers": 2,
            "num_filters": [32, 64],
            "stride": 1,
            "filter_size": 3,
            "activation": "relu"
        },
        "dense_layer": {
            "num_dense_layers": 1,
            "num_units": [64],
            "dropout": 0.5,
            "activation": "relu"
        },
        "output_layer": {
            "activation": "sigmoid"
        }
    }
}
