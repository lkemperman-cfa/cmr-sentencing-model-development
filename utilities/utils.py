import ast


def is_label_present(labels, label):
    return label in labels


def load_json_list(text):
    try:
        value = ast.literal_eval(text)
        return value
    except (ValueError, TypeError) as e:
        return []


def filter_labels(annotation, label_list):
    new_label = []
    for label in annotation:
        if (len(label["labels"]) > 0) and (label["labels"][0] in label_list):
            new_label.append(label)
        # elif len(label["labels"]) > 0:
        #     print(f"Did not find label {label['labels'][0]} in list")
    return new_label
