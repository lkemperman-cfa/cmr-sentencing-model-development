from enums.confidence import Confidence
from enums.days_range import DaysRange


def generate_range(min_value, max_value, source):
    return DaysRange(
        min_value=min_value,
        max_value=max_value,
        source=source
    ) if min_value is not None else None,


class AnalysisResponse:
    def __init__(
            self,
            sentence_id: str,
            confidence: Confidence = Confidence.LOW,
            fines_fees: bool = False,
            restitution: bool = False,
            life_sentence_indicator: bool = False,
            community_service: bool = False,
            diversion: bool = False,
            **kwargs
    ):
        # TODO: do we want to include the keys themselves in the response data if the classification is not there?
        self.response_data = {
            "id": sentence_id,
            "confidence": confidence.value,
            "confinement_range_days": generate_range(
                min_value=kwargs.get("minimum_confinement"),
                max_value=kwargs.get("maximum_confinement"),
                source=kwargs.get("confinement_source")
            ),
            "suspension_range_days": generate_range(
                min_value=kwargs.get("minimum_suspension"),
                max_value=kwargs.get("maximum_suspension"),
                source=kwargs.get("suspension_source")
            ),
            "probation_range_days": generate_range(
                min_value=kwargs.get("minimum_probation"),
                max_value=kwargs.get("maximum_probation"),
                source=kwargs.get("probation_source")
            ),
            "maximum_terms_of_confinement_days": kwargs.get("maximum_confinement"),
            "fines_fees": fines_fees,
            "restitution": restitution,
            "life_sentence_indicator": life_sentence_indicator,
            "community_service": community_service,
            "diversion": diversion
        }
