import pandas as pd
import json
import numpy as np

from json import JSONDecodeError
from utilities.utils import load_json_list, filter_labels


class BaseFormat:

    def __init__(self, entities: pd.DataFrame):
        self.entities = entities

    def format_label(self, **kwargs):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError


class LabelStudio(BaseFormat):

    def __init__(self, entities: pd.DataFrame):
        super().__init__(entities)

    @staticmethod
    def format_label(label_list, text):
        try:
            label_list = load_json_list(label_list)
            label_list = json.dumps(eval(label_list)) if isinstance(label_list, str) else label_list
            label_list_formatted = [
                {
                    "start": label["start"],
                    "end": label["end"],
                    "text": text,
                    "labels": label["labels"]
                }
                for label in label_list
            ]
        except (TypeError, JSONDecodeError) as e:
            print(f"Skipping formatting label={label_list}, exception={e}")
            label_list_formatted = [
                {
                    "start": pd.NA,
                    "end": pd.NA,
                    "text": text,
                    "labels": []
                }
            ]
        return label_list_formatted

    def get_labels(self):
        return self.entities.apply(
            lambda x: self.format_label(
                x["label"], x["Text"]
            ), axis=1
        )

    def load_labels(self, label_mappings):
        self.entities['label'] = (
            self.entities['label']
            .fillna("None")  # Replace pd.NA with "None"
            .replace({pd.NA: "None", np.nan: "None"})
            .str.replace("<NA>", "None", regex=False)  # Replace "nan" strings with "None"
            .str.replace("nan", "None", regex=False)
        )
        # label_mappings = self.csv_transformations.get("map_labels")
        if label_mappings:
            for key, value in label_mappings.items():
                self.entities["label"].str.replace(key, value, regex=False)

        self.entities["label"] = self.entities["label"].apply(load_json_list)
        self.entities['label'] = self.entities['label'].apply(
            lambda x: json.dumps(eval(x)) if isinstance(x, str) else x
        )


class AWSComprehend(BaseFormat):
    def format_label(self, begin_offset, end_offset, text):
        label_list = [
            {
                "start": begin_offset,
                "end": end_offset,
                "text": text,
                "labels": ["INCARCERATION"]  # TODO: load from entities
            }
        ]
        return label_list

    def get_labels(self):
        return self.entities.apply(
            lambda x: self.format_label(
                x["Begin Offset"], x["End Offset"], x["Text"]
            ), axis=1
        )
