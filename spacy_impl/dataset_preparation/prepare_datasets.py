import spacy
import os
import pandas as pd
import utilities.annotation_conversions as label_conversions
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans


def prepare_datasets_for_model(dataset_config: list, annotation_format: str):

    for dataset_to_convert in dataset_config:
        dataset_path = dataset_to_convert["dataset_path"]
        nlp = spacy.blank("en")
        doc_bin = DocBin()

        dataset = pd.read_csv(dataset_path)

        conversion_pipeline = getattr(label_conversions, annotation_format)(dataset)
        labels = conversion_pipeline.get_labels()

        for entities in tqdm(labels):
            if len(entities) > 0:
                text = entities[0]["text"]  # same for all entities
                if (text is None) or (pd.isna(text)):
                    continue
                doc = nlp.make_doc(text)
                ents = []

                for ent in entities:
                    start = ent["start"]
                    end = ent["end"]
                    if not pd.isna(start):
                        entity_label = ent["labels"][0]
                        span = doc.char_span(start, end, entity_label, alignment_mode="expand")
                        if span is not None:
                            ents.append(span)
                filtered_ents = filter_spans(ents)
                doc.ents = filtered_ents
                doc_bin.add(doc)

        # TODO: make directory for spaCy dataset
        output_file = dataset_to_convert["output_path"]
        spacy_dir = os.path.dirname(output_file)
        os.makedirs(spacy_dir, exist_ok=True)
        doc_bin.to_disk(output_file)
