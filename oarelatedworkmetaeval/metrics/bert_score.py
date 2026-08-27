import json

from oarelatedworkmetaeval.metrics.base import Metric
from oarelatedworkmetaeval.metrics.bertscorehfwrapper.bert_score import BertScore as HFBertScore


class BertScore(Metric):

    def __post_init__(self):
        self.bert_score = HFBertScore(model_type="microsoft/deberta-xlarge-mnli", lang="en",
                                      batch_size=2,
                                      rescale_with_baseline=True,
                                      rescale_to_zero_one=False)
    def __call__(
            self,
            target_paper_id: str,
            gen_model: str,
            summary: str,
            reference: str | None = None,
            target_paper: str | None = None,
            cited_papers: str | None = None) -> float:

        scores = self.bert_score.score(
            [summary],
            [target_paper + "\n\n" + cited_papers] if self.reference_free else [reference]
        )

        if self.sample_results_file is not None:
            with open(self.sample_results_file, 'a') as f:
                print(
                    json.dumps({
                        'target_paper_id': target_paper_id,
                        'model': gen_model,
                        'score': scores[2].item(),
                        'precision': scores[0].item(),
                        'recall': scores[1].item(),
                        'summary': summary,
                        'reference': target_paper + "\n\n" + cited_papers if self.reference_free else reference
                    }), file=f
                )
        # we select f1
        return scores[2].item()
