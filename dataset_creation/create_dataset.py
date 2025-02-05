import os
import pandas as pd

from utilities.utils import is_label_present


class BaseDatasetCreation:
    """Base class for creating new datasets for training and test

    Attributes
    ----------
    train_proportion : float
        a number between 0.0 and 1.0 to dictate the proportion of data allocated for the training split
    dataset_name: str
        name of dataset which dictates a folder to write all data and results to
    dataset_params: dict
        additional parameters specific to different subclasses
    """

    def __init__(self,
                 dataset_name: str,
                 full_df: pd.DataFrame,
                 train_proportion: float,
                 metadata: dict,
                 **kwargs
                 ):
        self.full_df = full_df
        self.dataset_name = dataset_name
        self.train_proportion = train_proportion
        self.metadata = metadata
        self.dataset_params = kwargs

    def generate_training_test_sets(self):
        raise NotImplementedError


class StratifiedSample(BaseDatasetCreation):
    """Strategy to equally represent different sources in training and test datasets

    Methods
    -------
    generate_training_test_sets()
        Divides the pool of data in the specified proportions into training and test, equally stratified by the
        `stratify` column parameter to represent those instances equally in training and test.

    """

    def generate_training_test_sets(self):
        stratify_column = self.dataset_params.get("stratify_column")  # TODO: allow for multiple columns?

        assert self.train_proportion <= 1.0, \
            f"Train proportion must be less than or equal to 1.0. It is {self.train_proportion}"
        if stratify_column == "label":
            print("stratifying by label")
            all_labels = self.dataset_params.get("label_list")
            stratify_column = []
            for label in all_labels:
                column_name = f"{label}_present"
                self.full_df[column_name] = self.full_df.apply(
                    lambda x: is_label_present(x["label"], label), axis=1
                )
                stratify_column.append(column_name)
        training_data = self.full_df.groupby(stratify_column, group_keys=False).apply(
            lambda x: x.sample(frac=self.train_proportion)
        )
        self.metadata["train"]["size"] = len(training_data)
        test_data = self.full_df[~self.full_df.isin(training_data)].dropna()
        self.metadata["test"]["size"] = len(test_data)
        dataset_path = os.path.join("datasets/", self.dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        training_data.to_csv(os.path.join(dataset_path, "train_df.csv"))
        test_data.to_csv(os.path.join(dataset_path, "test_df.csv"))
