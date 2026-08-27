
from typing import Sequence, Union, Tuple, Optional

# The dependencies in https://github.com/google-research/google-research/blob/master/rouge/requirements.txt
import datasets
import evaluate
import torch
from bert_score import BERTScorer
from datasets import Column

_CITATION = """\
@inproceedings{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkeHuCVFDr}
}
"""

_DESCRIPTION = """\
BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference
sentences by cosine similarity.
It has been shown to correlate with human judgment on sentence-level and system-level evaluation.
Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different language
generation tasks.
See the project's README at https://github.com/Tiiiger/bert_score#readme for more information.
"""

_KWARGS_DESCRIPTION = """"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BertScore(evaluate.Metric):

    def __init__(self, device='cuda:0', model_type="microsoft/deberta-xlarge-mnli", lang="en", batch_size: int = 4,
                 rescale_with_baseline: bool = False, aggregate: bool = False, model: Optional[BERTScorer] = None, **kwargs):
        super().__init__(**kwargs)
        self.verbose = kwargs.get("verbose", True)

        if model is not None:
            self.model = model
        else:
            self.model = BERTScorer(model_type=model_type, lang=lang, rescale_with_baseline=rescale_with_baseline,
                                    device=device)

        if self.model._tokenizer.model_max_length != self.model._model.config.max_position_embeddings:
            self.model._tokenizer.model_max_length = self.model._model.config.max_position_embeddings

        self.batch_size = batch_size
        self.aggregate = aggregate

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string"),
                        "references": datasets.Sequence(datasets.Value("string")),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string"),
                        "references": datasets.Value("string"),
                    }
                ),
            ],
        )

    def _compute(self, predictions, references):
        if isinstance(predictions, Column):
            predictions = list(predictions)
        if isinstance(references, Column):
            references = list(references)

        res = self.model.score(predictions, references, batch_size=self.batch_size)

        precision, recall, f1 = res
        if precision.shape[0] > 1 and self.aggregate:
            precision = precision.mean().item()  # take sample mean of f1 score
            recall = recall.mean().item()
            f1 = f1.mean().item()
        else:
            # convert to list of floats
            precision = precision.tolist()
            recall = recall.tolist()
            f1 = f1.tolist()

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def score(self, src: Sequence[str], tgts: Sequence[Sequence[str]]) -> Tuple[
        torch.tensor, torch.tensor, torch.tensor]:
        """
        Computes similarity of source and target sequence.

        Args:
            src: Source sequence.
            tgts: References of target sequence.

        Returns:
            Tuple of precision, recall and f1 score.
                each is of shape (N); N = number of input sources
        """
        return self.model.score(src, tgts, batch_size=self.batch_size)
