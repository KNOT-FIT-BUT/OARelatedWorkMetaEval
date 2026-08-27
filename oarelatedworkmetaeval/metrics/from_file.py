import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from classconfig import ConfigurableValue, RelativePathTransformer
from datasets import load_from_disk, load_dataset

from oarelatedworkmetaeval.metrics.base import Metric


class RelativePathTransformerList(RelativePathTransformer):

    def __call__(self, path: list[Optional[str]]) -> list[Optional[str]]:
        return [super().__call__(p) for p in path]


class FromFileCoarseScore(Metric):
    """
    It is not computing the metric, it is just loading results from given file.
    """

    score_file: list[str] = ConfigurableValue(
        desc='Loads scores for each summary from given files and averages the scores when more than one file is provided. It expects jsonl with fields: target_paper_id, model, score',
        transform=RelativePathTransformerList(allow_none=False)
    )

    def __post_init__(self):
        self.scores = defaultdict(float)
        file_path = None
        try:
            for file_path in self.score_file:
                with open(file_path, 'r') as f:
                    scores = {}
                    for line in f:
                        data = json.loads(line)
                        scores[(int(data['target_paper_id']), data['model'])] = data['score']

                if len(self.scores) != 0 and len(self.scores) != len(scores):
                    raise ValueError(f"Number of scores in file {file_path} does not match the number of scores in previous files")

                for k, v in scores.items():
                    self.scores[k] += v

            for k in self.scores:
                self.scores[k] /= len(self.score_file)
        except Exception as e:
            raise ValueError(
                f"Malformed score file '{file_path}': {e}. "
                f"Expected jsonl with fields: target_paper_id, model, score"
            ) from e

    def __call__(self,
                 target_paper_id: int,
                 gen_model: str,
                 summary: str,
                 reference: str | None = None,
                 target_paper: str | None = None,
                 cited_papers: str | None = None) -> float:

        if self.sample_results_file is not None:
            with open(self.sample_results_file, 'a') as f:
                print(
                    json.dumps({
                        'target_paper_id': target_paper_id,
                        'model': gen_model,
                        'score': self.scores[(target_paper_id, gen_model)],
                        'summary': summary,
                        'reference': target_paper + "\n\n" + cited_papers if self.reference_free else reference
                    }), file=f
                )

        return self.scores[(target_paper_id, gen_model)]


class FromFileStatementLevel2CoarseScore(Metric):
    """
    It is converting the statement level prediction into coarse scores.
    """

    dataset_path: Optional[str] = ConfigurableValue(desc="Dataset path or None to load it from HF hub. It uses the statements config",
                                                    user_default=None,
                                                    transform=RelativePathTransformer(allow_none=True))

    statement_cls_file: list[str] = ConfigurableValue(
        desc="List of files containing classification results for each statement, if more than one is provided the score will be averaged",
        transform=RelativePathTransformerList(allow_none=False)
    )

    def __post_init__(self):
        # load data
        dataset_path = Path(self.dataset_path)
        if dataset_path.exists():
            self.dataset = load_from_disk(dataset_path)
        else:
            self.dataset = load_dataset(
                "BUT-FIT/OARelatedWorkMetaEval",
                "statements",
                split="train"
            )

        self.id_2_rw = {}

        for r in self.dataset:
            parts = r["statement_id"].split("_", maxsplit=1)
            paper_id, model = parts[0], parts[1]
            model = "_".join(model.split("_")[:-1])
            self.id_2_rw[int(r["id"])] = tuple([int(paper_id), model])

        # transform scores
        self.scores = defaultdict(float)

        for f_path in self.statement_cls_file:
            scores = defaultdict(float)
            scores_total = defaultdict(float)
            with open(f_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    rw_id = self.id_2_rw[int(data["id"])]

                    scores[(int(rw_id[0]), rw_id[1])] += data["classification"].strip().lower() == "true"
                    scores_total[(int(rw_id[0]), rw_id[1])] += 1.0

            if len(self.scores) != 0 and len(self.scores) != len(scores):
                raise ValueError(f"Number of scores in file {f_path} does not match the number of scores in previous files")

            for k, total in scores_total.items():
                self.scores[k] += scores[k] / total

        for k in self.scores:
            self.scores[k] /= len(self.statement_cls_file)

    def __call__(self,
                 target_paper_id: int,
                 gen_model: str,
                 summary: str,
                 reference: str | None = None,
                 target_paper: str | None = None,
                 cited_papers: str | None = None) -> float:

        if self.sample_results_file is not None:
            with open(self.sample_results_file, 'a') as f:
                print(
                    json.dumps({
                        'target_paper_id': target_paper_id,
                        'model': gen_model,
                        'score': self.scores[(target_paper_id, gen_model)],
                        'summary': summary,
                        'reference': target_paper + "\n\n" + cited_papers if self.reference_free else reference
                    }), file=f
                )

        return self.scores[(target_paper_id, gen_model)]
