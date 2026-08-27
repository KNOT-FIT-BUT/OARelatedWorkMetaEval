from classconfig import ConfigurableValue

from oarelatedworkmetaeval.metrics.base import Metric
from enum import Enum

from oarelatedworkmetaeval.metrics.hf_block_matching import AParBlockMatchingMetric
from oarelatedworkmetaeval.metrics.huggingfacerouge.rouge import Rouge as HFRouge
from oarelatedworkmetaeval.metrics.bertscorehfwrapper.bert_score import BertScore as BertScoreHFWrapper


class BaseMetric(Enum):
    ROUGE_2_F1 = "rouge_2_f1"
    BERT_SCORE = "bert_score"


class BlockMatching(Metric):
    """
    Meta metric for calculating similarity of blocks.

    It is comparing best matching blocks using given metric.

    It is called meta metric because it is using other metrics operating on plain string level,
    but upgrades it to block level.
    """

    metric: BaseMetric = ConfigurableValue(
        desc="Base metric for block matching. It is expected to be a metric operating on plain string level.",
        user_default=BaseMetric.ROUGE_2_F1.name,
    )

    def __post_init__(self):
        if self.metric == BaseMetric.ROUGE_2_F1.name:
            rouge = HFRouge(allow_aggregate=False, verbose=False)
            self.block_match = AParBlockMatchingMetric(
                metric=rouge,
                sel_key="rouge2_fmeasure",
                artificial_paragraphs=0,
                verbose=False
            )
        elif self.metric == BaseMetric.BERT_SCORE.name:
            self.block_match = AParBlockMatchingMetric(
                metric=BertScoreHFWrapper(model_type="microsoft/deberta-xlarge-mnli", lang="en",
                                          batch_size=2,
                                          rescale_with_baseline=True,
                                          rescale_to_zero_one=False),
                sel_key="f1",
                artificial_paragraphs=0,
                verbose=False
            )
        else:
            raise ValueError(f"Unsupported metric {self.metric} for block matching")

    def __call__(self,
                 target_paper_id: str,
                 gen_model: str,
                 summary: str,
                 reference: str | None = None,
                 target_paper: str | None = None,
                 cited_papers: str | None = None) -> float:

        score = self.block_match.score(
            summary,
            [target_paper + "\n\n" + cited_papers] if self.reference_free else [reference]
        )

        if self.sample_results_file is not None:
            with open(self.sample_results_file, 'a') as f:
                print(
                    {
                        'target_paper_id': target_paper_id,
                        'model': gen_model,
                        'score': score,
                        'summary': summary,
                        'reference': target_paper + "\n\n" + cited_papers if self.reference_free else reference
                    },
                    file=f
                )

        return score
