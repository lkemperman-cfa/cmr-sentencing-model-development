class BaseAnalyzer:
    def __init__(self, sentences: list):
        self.sentences = sentences

    def analyze(self):
        raise NotImplementedError
