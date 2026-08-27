from abc import ABC, abstractmethod

from classconfig import ConfigurableMixin, ConfigurableValue, RelativePathTransformer
from classconfig.validators import BoolValidator, AnyValidator, StringValidator


class Metric(ABC, ConfigurableMixin):
    name: str = ConfigurableValue(desc='Name of metric')
    sample_results_file: str | None = ConfigurableValue(
        desc='Path to file with detailed results for each sample in jsonl format',
        user_default=None,
        transform=RelativePathTransformer(allow_none=True),
    )
    reference_free: bool | str = ConfigurableValue(
        desc='Whether the metric is reference free or reference based',
        validator=AnyValidator([BoolValidator(), lambda x: x == "go"])
    )
    skip_humans: bool = ConfigurableValue(
        desc='Whether the human related works should be skipped',
        user_default=False,
        validator=BoolValidator(),
        voluntary=True
    )

    @abstractmethod
    def __call__(self,
                 target_paper_id: str,
                 gen_model: str,
                 summary: str,
                 reference: str | None = None,
                 target_paper: str | None = None,
                 cited_papers: str | None = None) -> float:
        """
        Perform a metric evaluation

        :param target_paper_id: id of the target paper for which the summary was generated for
        :param gen_model: name of model that generated the summary
        :param summary: generated summary that is supposed to be evaluated
        :param reference: voluntary reference used for evaluation in case of reference based setup
        :param target_paper: rest of target paper, without related work section, for which the summary was generated for
            used for evaluation in case of reference free setup
        :param cited_papers: cited papers content
            used for evaluation in case of reference free setup
        :return: metric score for given summary
        """
        ...

