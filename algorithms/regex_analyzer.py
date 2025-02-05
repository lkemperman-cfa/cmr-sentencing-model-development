from __future__ import annotations

from algorithms.analysis_response import AnalysisResponse
from algorithms.base_analyzer import BaseAnalyzer
from utilities import utils
import re
import string

RGX = re.compile(r"""
         (?P<YEARS>\d+\s?Y[EARS]*)?  # YEARS
         (?:[\s\-\,\&\+]|and)*
         (?P<MONTHS>\d+\s?M[ONTHS]*)?    # MONTHS
         (?:[\s\-\,\&\+]|and)*
          (?P<WEEKS>\d+\s?W[EKS]+)?    # WEEKS
         (?:[\s\-\,\&\+]|and)*
         (?P<DAYS>\d{1,2}\s?D[AYS]*)?    # DAYS
         (?:[\s\-\,\&\+]|and)*
         (?P<HOURS>\d{1,2}\s?H[OURS]*)?  #HOUR
         |(?<![\$\d]\s)(?<![\$\d])(?P<MONTHS2>11)(?:[\s\-\/]+)(?P<DAYS2>[123]?\d)\s(?!YE|MO|DA)  #ALT MO/DY
        """, re.VERBOSE + re.IGNORECASE)

MULTIPLIERS = dict([
    ('YEARS', 365.25),
    ('MONTHS', 30.4375),
    ('WEEKS', 7.0),
    ('DAYS', 1.0),
    ('HOURS', 0.04167)
])


def convert_to_digits(text):
    """Takes string and finds/converts alpha integers into digits"""
    words_to_numbers = {
        'ninety': '90',
        'eighty': '80',
        'seventy': '70',
        'sixty': '60',
        'fifty': '50',
        'forty': '40',
        'thirty': '30',
        'twenty': '20',
        'nineteen': '19',
        'eighteen': '18',
        'seventeen': '17',
        'sixteen': '16',
        'fifteen': '15',
        'fourteen': '14',
        'thirteen': '13',
        'twelve': '12',
        'eleven': '11',
        'ten': '10',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'zero': '0'
    }
    pattern = re.compile(r'\b(' + '|'.join(words_to_numbers.keys()) + r')\b', re.IGNORECASE)
    return re.sub(pattern, lambda x: words_to_numbers[x.group().lower()], text)


def calculate_days(units):
    """Estimates total days from duration units"""
    days = 0
    for key, val in units.items():
        days = days + (MULTIPLIERS[re.sub(r'[\d]', '', key)] * float(val))
    return int(round(days))


def classify_type(text):
    """Classifies the type of sentence"""
    substring_to_response_keys = {
        'JAIL': ['maximum_confinement', 'minimum_confinement'],  # for now, setting min and max to the same value
        'CONF': ['maximum_confinement', 'minimum_confinement'],
        'PROB': ['probation_range'],
        'SUSP': ['suspension_range']
    }
    # TODO: should we consider multiple matches or just first? I'll stick with first for now for simplicity.
    for search_key, value in substring_to_response_keys.items():
        search_result = re.search(search_key, text)  # finds first match and then returns result
        if search_result is not None:
            return value


def process_adj_text(text, position='pre', max_length=10):
    """returns a string of adjacent text before or after a duration"""
    split_text = re.split(r'[^\w\s]', text)
    if position == 'pre':
        text = split_text[-1][-max_length:].strip() if len(split_text) > 0 else text[-max_length:]
    else:
        text = split_text[0][0:max_length].strip() if len(split_text) > 0 else text[0:max_length].strip()
    # remove punctuation and extra spaces
    return utils.cleanup_phrase(re.sub(r'\s+', ' ', text.translate(str.maketrans('', '', string.punctuation))))


def get_durations(text):
    """Takes a string and returns list of dictionaries describing durations"""
    results = []
    phrases = []

    matches = RGX.finditer(text.upper())

    for i in matches:
        for j in range(len(i.groups())):
            if i.groups()[j] is not None and utils.cleanup_phrase(i.group()) not in phrases:
                phrases.append(utils.cleanup_phrase(i.group()))
                units = {}
                pre_text = process_adj_text(text.upper()[max(0, i.span()[0] - 12):i.span()[0]], 'pre')
                post_text = process_adj_text(text.upper()[i.span()[1]:min(len(text.upper()), i.span()[1] + 12)], 'post')
                for key, value in i.groupdict().items():
                    if value is not None:
                        int_match = re.search(r'\d+', value)
                        int_value = int(int_match.group(0) if int_match else 0)
                        units[re.sub(r'[\d]', '', key)] = int_value

                days = calculate_days(units)

                results.append({
                    'days': days if days > 0 else None,
                    'units': units,
                    'text': utils.cleanup_phrase(i.group()),
                    'pre_text': pre_text,
                    'post_text': post_text
                })

    return results


def duration_search(text: str = '') -> list:
    """Public function returns a list of durations from a string"""
    text = convert_to_digits(text)
    results = get_durations(text)
    return results


class RegexAnalyzer(BaseAnalyzer):
    def __init__(self, sentences):
        super().__init__(sentences)

    def analyze(self):
        responses = []
        for sentence in self.sentences:
            analysis_response_dict = {}
            results = duration_search(sentence["text"])
            classification_results = classify_type(sentence["text"])
            for key in classification_results:
                # TODO: for now assuming single result, need to update to handle multiple
                #  / or case when results is empty
                analysis_response_dict[key] = results[0]['days']
            responses.append(
                AnalysisResponse(
                    sentence_id=sentence["id"],
                    **analysis_response_dict).response_data
            )
        return responses
