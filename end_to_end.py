import os
import json
import inquirer
from inquirer import errors

from spacy_impl.dataset_preparation import prepare_datasets
from utilities.annotation_conversions import BaseFormat
from dataset_creation.create_dataset import BaseDatasetCreation, StratifiedSample
from dataset_creation.dataset_processing import DatasetProcessor
from spacy_impl.train import train_model
from spacy_impl.evaluate import evaluate


def create_experiment_start_qs():
    questions = [
        inquirer.Text(
            "experiment_name", message="Please set a descriptive name for your experiment", default=""
        ),
        inquirer.List(
            "dataset_creation", message="Do you want to use existing training and test datasets, or create new ones?",
            choices=["use existing", "create new"]
        )
    ]
    return inquirer.prompt(questions)


def get_root_directories():
    return [d for d in os.listdir(".") if os.path.isdir(d) and d not in [".git", ".idea", "raw_data"]]


def use_existing_datasets_qs():
    questions = [
        inquirer.List(
            "dataset_directory", message="Please select the directory where your datasets live",
            choices=get_root_directories(),
            default="datasets"
        )
    ]

    experiment_info = inquirer.prompt(questions)

    dataset_directory = experiment_info["dataset_directory"]

    def validate_experiment_path(answers, current):
        # if not os.path.exists(os.path.join(experiment_directory, current, "datasets/")):
        #     raise errors.ValidationError('', reason="Could not find expected datasets in the provided experiment path")
        return True

    experiment_location_questions = [
        inquirer.List("dataset_name", message="Please select the dataset you'd like to use",
                      choices=[d for d in os.listdir(dataset_directory)], validate=validate_experiment_path)
    ]
    dataset_name = inquirer.prompt(experiment_location_questions)["dataset_name"]
    return os.path.join(dataset_directory, dataset_name)


def get_all_labels():
    # TODO: actually load these from the data?
    return [
        "CONFINEMENT",
        "CONFINEMENT_DURATION",
        "PROBATION",
        "PROBATION_DURATION",
        "MONETARY_PENALTY",
        "MONETARY_PENALTY_AMOUNT",
        "CONDITIONAL",
        "OTHER_PUNISHMENT",
        "SENTENCE",
        "SENTENCE_DATE"
    ]


def get_columns_to_stratify():
    # TODO: same - load from data
    return ["Source", "label"]


def load_default_config(config_filepath):
    with open(config_filepath) as config_file:
        config_data = json.load(config_file)
    return config_data


def stratified_sample_creation_qs():
    experiment_questions = [
        inquirer.Text("dataset_name", message="Create a descriptive name for your dataset"),
        inquirer.Checkbox("all_datasets", message="Please select the raw data to use for your dataset",
                          choices=os.listdir("datasets/raw_data/")),
        inquirer.Checkbox("label_list", message="Please select the labels to include for your experiment",
                          choices=get_all_labels()),
        inquirer.Text("train_proportion", message="Please select the proportion of the total dataset to use "
                                                  "for training (from 0.0 to 1.0)",
                      validate=lambda _, c: 0.0 <= float(c) <= 1.0),
        inquirer.List("stratify_column", message="Please choose a property to use for stratifying the data equally "
                                                 "between training and test", choices=get_columns_to_stratify())
    ]
    results = inquirer.prompt(experiment_questions)
    results["json_filenames"] = [
        os.path.join("datasets", "raw_data", f) for f in results["all_datasets"] if f.endswith("json")
    ]
    results["csv_filenames"] = [
        os.path.join("datasets", "raw_data", f) for f in results["all_datasets"] if f.endswith("csv")
    ]
    results["train_proportion"] = float(results["train_proportion"])
    del results["all_datasets"]
    return results


def prepare_dataset_qs():
    label_format_options = [cls.__name__ for cls in BaseFormat.__subclasses__()]
    questions = [
        inquirer.List('annotation_format',
                      message="Select the format of labels that your dataset uses",
                      choices=label_format_options,
                      )
    ]
    return inquirer.prompt(questions)


def create_experiment_directories(experiment_path):
    [os.makedirs(os.path.join(experiment_path, folder)) for folder in ["models", "results"]]


def end_to_end():
    experiment_data = create_experiment_start_qs()
    experiment_name = experiment_data["experiment_name"]
    experiment_path = os.path.join("experiments", experiment_name)
    create_experiment_directories(experiment_path)
    if experiment_data["dataset_creation"] == "use existing":
        dataset_path = use_existing_datasets_qs()
    else:
        dataset_creation_options = [cls.__name__ for cls in BaseDatasetCreation.__subclasses__()]
        questions = [
            inquirer.List('dataset_creation_method',
                          message="Select a method to create your dataset from the available methods",
                          choices=dataset_creation_options,
                          ),
        ]
        answers = inquirer.prompt(questions)
        if answers["dataset_creation_method"] == "StratifiedSample":
            sample_config = stratified_sample_creation_qs()
            dataset_path = os.path.join("datasets", sample_config["dataset_name"])
            dataset_config = load_default_config('dataset_creation/dataset_creation_config.json')
            dataset_config.update(sample_config)
            dataset_metadata = {
                "label_counts": {},
                "train": {
                    "sources": {},
                    "label_counts": {}
                },
                "test": {
                    "sources": {},
                    "label_counts": {}
                }
            }
            processor = DatasetProcessor(
                **dataset_config,
                annotation_format="LabelStudio",
                metadata=dataset_metadata
            )
            dataset = processor.get_full_dataset()
            creator = StratifiedSample(
                full_df=dataset,
                annotation_format="LabelStudio",
                metadata=dataset_metadata,
                **dataset_config
            )
            creator.generate_training_test_sets()
            dataset_metadata = os.path.join(dataset_path, "metadata.json")
            dataset_config.update(creator.metadata)
            with open(dataset_metadata, "w") as outfile:
                json.dump(dataset_config, outfile, indent=4)
            # prepare datasets for spaCy
            dataset_preparation_config = prepare_dataset_qs()
            dataset_preparation_config["dataset_config"] = [
                {
                    "dataset_path": os.path.join(dataset_path, "train_df.csv"),
                    "output_path": os.path.join(dataset_path, "spacy", "train.spacy")
                },
                {
                    "dataset_path": os.path.join(dataset_path, "test_df.csv"),
                    "output_path": os.path.join(dataset_path, "spacy", "test.spacy")
                }
            ]
            prepare_datasets.prepare_datasets_for_model(**dataset_preparation_config)
        else:
            raise NotImplementedError

    train_model(dataset_path, experiment_path)
    evaluate(dataset_path, experiment_path)


if __name__ == "__main__":
    end_to_end()
