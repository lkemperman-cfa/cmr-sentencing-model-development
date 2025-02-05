import json
import statistics

import pandas as pd
import utilities.annotation_conversions as label_conversions
from utilities.utils import filter_labels


class DatasetProcessor:

    def __init__(self,
                 train_proportion: float,
                 csv_filenames: list,
                 json_filenames: list,
                 csv_transformations: dict,
                 json_transformations: dict,
                 columns: list,
                 label_list: list,
                 dataset_name: str,
                 annotation_format: str,
                 metadata: dict,
                 **kwargs
                 ):
        self.train_proportion = train_proportion
        self.csv_filenames = csv_filenames
        self.csv_transformations = csv_transformations
        self.json_transformations = json_transformations
        self.json_filenames = json_filenames
        self.columns = columns
        self.label_list = label_list
        self.dataset_name = dataset_name
        self.annotation_format = annotation_format
        self.dataset_params = kwargs
        self.metadata = metadata

    def get_full_dataset(self):
        df_from_csvs = self.process_csvs()
        df_from_jsons = self.process_jsons()
        full_df = pd.concat([df_from_csvs, df_from_jsons], ignore_index=True)
        self.get_label_counts(full_df)
        full_df["label"] = full_df["label"].apply(lambda x: filter_labels(x, label_list=self.label_list))
        return full_df

    def get_label_counts(self, df):
        label_stats = {"num_labels": [], "label_statistics": {}}
        summary_statistics = {}
        for index, row in df.iterrows():
            labels = row["label"]
            overall_counts = len(labels)
            all_labels = [label["labels"][0] if label["labels"] else None for label in labels]

            unique_labels = set(all_labels)
            label_counts = {item: all_labels.count(item) for item in unique_labels}
            for key, value in label_counts.items():
                if key in label_stats["label_statistics"].keys():
                    label_stats["label_statistics"][key].append(value)
                else:
                    label_stats["label_statistics"][key] = [value]
            label_stats["num_labels"].append(overall_counts)
        summary_statistics["max_labels_per_row"] = max(label_stats["num_labels"])
        summary_statistics["avg_labels_per_row"] = statistics.mean(label_stats["num_labels"])
        summary_statistics["med_labels_per_row"] = statistics.median(label_stats["num_labels"])
        for label, values in label_stats["label_statistics"].items():
            summary_statistics[label] = {}
            summary_statistics[label]["rows_with_label"] = len(values)
            summary_statistics[label]["max_instances_per_row"] = max(values)
        self.metadata.update(summary_statistics)

    def process_csvs(self):
        df = pd.concat(
            [
                pd.read_csv(csv, usecols=self.columns, header=0) for csv in self.csv_filenames
            ],
            ignore_index=True, join='inner'
        )
        self.metadata["num_csvs"] = len(self.csv_filenames)  # metadata for number of csvs
        conversion_pipeline = getattr(label_conversions, self.annotation_format)(df)
        conversion_pipeline.load_labels(
            label_mappings=self.csv_transformations.get("map_labels")
        )
        return conversion_pipeline.entities

    def process_jsons(self):
        mapping_params = self.dataset_params.get("map_labels", {})
        merged_json = pd.DataFrame(columns=self.columns)
        self.metadata["num_jsons"] = len(self.json_filenames)  # metadata for number of csvs
        for json_path in self.json_filenames:
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                for i in range(len(data)):
                    row_metadata = data[i]["data"]
                    row = {"Source": row_metadata["Source"], "Text": row_metadata["Data"], "label": []}
                    text = row_metadata["Data"]
                    for j in range(len(data[i]["annotations"])):
                        annotations = data[i]["annotations"][j]["result"]
                        for k in range(len(annotations)):
                            if "value" in annotations[k].keys():
                                label_info = annotations[k]["value"]
                                labels = label_info["labels"]
                                label_value = label_info["labels"][0]

                                if label_value in mapping_params.keys():
                                    labels = [mapping_params.get(label_value)]
                                label_value = labels[0]
                                row["label"].append(
                                    {
                                        "start": label_info["start"],
                                        "end": label_info["end"],
                                        "text": text,
                                        "labels": [label_value]
                                    }
                                )
                    merged_json.loc[len(merged_json)] = row
        return merged_json
