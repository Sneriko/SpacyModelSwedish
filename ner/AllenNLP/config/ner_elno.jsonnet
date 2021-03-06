{
    "vocabulary": {
        "pretrained_files": {"tokens": "/content/drive/My Drive/FastTextVectors/cc.sv.300.vec"},
        "min_pretrained_embeddings": {"tokens": 500000}
    },
    "dataset_reader": {
        "type": "conll2003",
        "coding_scheme": "BIOUL",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            },
            "token_characters": {
                "type": "characters"
            },
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "train_data_path": "/content/drive/My Drive/Examensarbete/Data_suc/SUC3trainCONLL2003",
    "validation_data_path": "/content/drive/My Drive/Examensarbete/Data_suc/SUC3testCONLL2003",
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 64
        }
    },
    "model": {
        "type": "crf_tagger",
        "dropout": 0.5,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 200,
            "input_size": 1452,
            "num_layers": 2,
            "recurrent_dropout_probability": 0.5,
            "use_highway": true
        },
        "feedforward": {
            "activations": "tanh",
            "dropout": 0.5,
            "hidden_dims": 400,
            "input_dim": 400,
            "num_layers": 1
        },
        "include_start_end_transitions": false,
        "initializer": {
          "regexes": [
            [
                ".*tag_projection_layer.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*tag_projection_layer.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*feedforward.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*feedforward.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*weight_ih.*",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*weight_hh.*",
                {
                    "type": "orthogonal"
                }
            ],
            [
                ".*bias_ih.*",
                {
                    "type": "zero"
                }
            ],
            [
                ".*bias_hh.*",
                {
                    "type": "lstm_hidden_bias"
                }
            ]
          ]
        },
        "label_encoding": "BIOUL",
        "regularizer": {
          "regexes": [
            [
                "scalar_parameters",
                {
                    "alpha": 0.001,
                    "type": "l2"
                }
            ]
          ]
        },
        "text_field_embedder": {
          "token_embedders": {
            "elmo": {
                "type": "elmo_token_embedder",
                "options_file": "/content/drive/My Drive/elmo/swedish/options.json",
                "weight_file": "/content/drive/My Drive/elmo/swedish/swedish-elmo-weights.hdf5",
                "do_layer_norm": false,
                "dropout": 0
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": 25,
                    "sparse": true,
                    "vocab_namespace": "token_characters"
                },
                "encoder": {
                    "type": "lstm",
                    "hidden_size": 128,
                    "input_size": 25,
                    "num_layers": 1
                }
            },
            "tokens": {
                "type": "embedding",
                "embedding_dim": 300,
                "pretrained_file": "/content/drive/My Drive/FastTextVectors/cc.sv.300.vec",
                "sparse": true,
                "trainable": true
            }
          }
        },
        "verbose_metrics": true
    },
    "trainer": {
        "cuda_device": 0,
        "grad_norm": 5,
        "num_epochs": 30,
        "optimizer": {
            "type": "dense_sparse_adam",
            "lr": 0.001
        },
        "patience": 25,
        "validation_metric": "+f1-measure-overall"
    }
}