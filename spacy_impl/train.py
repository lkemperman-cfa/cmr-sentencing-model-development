import spacy
import os
from spacy.cli.train import train


def generate_config():
    spacy.cli.init_config(  # Path to save the configuration file
        lang="en",  # Language
        pipeline=["ner"],  # Pipeline components
        optimize="accuracy"  # Optimization mode (optional, "accuracy" or "efficiency")
    ).to_disk("spacy/training_config/config.cfg")


def train_model(dataset_path: str, experiment_path: str, write_config: bool = False):
    if write_config:
        generate_config()

    spacy.cli.download("en_core_web_lg")
    model_store_directory = os.path.join(experiment_path, "models")
    train_dataset_path = os.path.join(dataset_path, "spacy", "train.spacy")

    # TODO: externalize these configurations and expose more customizations
    train(
        "spacy_impl/training_config/config.cfg", model_store_directory,
        overrides={
            "paths.train": train_dataset_path,
            "paths.dev": train_dataset_path
        }
    )


# if __name__ == "__main__":
#     train_model(write_config=False)
