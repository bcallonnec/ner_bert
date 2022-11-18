"""
Predict using model trained

Be carefull to set the following environments variables (in bashrc for instance):
- CUDA_VISIBLE_DEVICES
- TOKENIZERS_PARALLELISM
"""
from typing import Any

import hydra

from src.model.datasets import NERDataset


@hydra.main(config_path="conf", config_name="predict", version_base=hydra.__version__)
def main(cfg: Any) -> None:
    """
    Main function that train the model
    """
    if cfg.test_mode:
        cfg.data["split"] = "test[:20]"

    # Load train data
    data_test = hydra.utils.instantiate(cfg.data, _convert_="all")

    # Instantiate tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer, _convert_="all")

    # Construct Dataset
    dataset = NERDataset(
        texts=data_test["tokens"],
        labels=data_test["ner_tags"],
        tokenizer=tokenizer,
        max_len=cfg.seq_max,
        loss_ignore_index=cfg.loss_ignore_index,
        propagate_label_to_word_pieces=cfg.propagate_label_to_word_pieces,
    )

    # Construct DataLoader
    data_loader = dataset.get_data_loader(
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_data,
        num_workers=cfg.num_data_workers,
    )

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, _convert_="all")

    # Predict
    preds, probas = model.predict(data_loader)

    print(preds)
    print(probas)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
