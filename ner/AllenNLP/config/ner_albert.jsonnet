local transformer_model = "KB/bert-base-swedish-cased";
local transformer_hidden_dim = 768;
local epochs = 25;
local batch_size = 8;
local max_length = 512;

{
    "dataset_reader": {
        "type": "conll2003",
        "tag_label": "ner",
        "coding_scheme": "BIOUL",
        "token_indexers": {
          "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": "KB/bert-base-swedish-cased-ner",
            "max_length": 512
          },
        },
    },
    "train_data_path": "/content/drive/My Drive/Examensarbete/Data_suc/SUC3trainCONLL2003",
    "validation_data_path": "/content/drive/My Drive/Examensarbete/Data_suc/SUC3testCONLL2003",
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 8
        }
    },
    "model": {
        "type": "crf_tagger",
        "encoder": {
            "type": "pass_through",
            "input_dim": 768,
        },
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "text_field_embedder": {
          "token_embedders": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": "KB/bert-base-swedish-cased-ner",
                "max_length": 512,
            }
          }
        },
        "verbose_metrics": true
    },
    "trainer": {
        "optimizer": {
          "type": "huggingface_adamw",
          "weight_decay": 0.0,
          "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
          "lr": 1e-5,
          "eps": 1e-8
        },
        "learning_rate_scheduler": {
          "type": "slanted_triangular",
          "cut_frac": 0.05,
        },
        "grad_norm": 1.0,
        "num_epochs": 10,
        "patience": 5,
        "cuda_device": 0,
        "validation_metric": "+f1-measure-overall"
    }
}