import spacy
import os
import json
from spacy.tokens import DocBin
from spacy.training.example import Example


def evaluate(dataset_path: str, experiment_path: str):
    model_path = os.path.join(experiment_path, "models", "model-best")
    nlp = spacy.load(model_path)
    test_dataset_path = os.path.join(dataset_path, "spacy", "test.spacy")
    doc_bin = DocBin().from_disk(test_dataset_path)
    test_docs = list(doc_bin.get_docs(nlp.vocab))

    examples = [Example(predicted=nlp(doc.text), reference=doc) for doc in test_docs]

    results = nlp.evaluate(examples)
    results_file = os.path.join(experiment_path, "results", "results.json")
    with open(results_file, "w") as outfile:
        json.dump(results, outfile, indent=4)
