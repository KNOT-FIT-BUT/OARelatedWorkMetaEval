# I've used implementation from huggingface and altered the ROUGE-L-SUM  as there was no option for sentence splitting.
# also only the fmeasure is reported, but I want the full experience!
# Copyright 2020 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ROUGE metric from Google Research github repo. """

# The dependencies in https://github.com/google-research/google-research/blob/master/rouge/requirements.txt
import absl  # Here to have a nice missing dependency error message early on
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy  # Here to have a nice missing dependency error message early on
import six  # Here to have a nice missing dependency error message early on
from rouge_score import rouge_scorer, scoring

import evaluate
from tqdm import tqdm

_CITATION = """\
@inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W04-1013",
    pages = "74--81",
}
"""

_DESCRIPTION = """\
ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for
evaluating automatic summarization and machine translation software in natural language processing.
The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.

Note that ROUGE is case insensitive, meaning that upper case letters are treated the same way as lower case letters.

This metrics is a wrapper around Google Research reimplementation of ROUGE:
https://github.com/google-research/google-research/tree/master/rouge
"""

_KWARGS_DESCRIPTION = """
Calculates average rouge scores for a list of hypotheses and references
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    rouge_types: A list of rouge types to calculate.
        Valid names:
        `"rouge{n}"` (e.g. `"rouge1"`, `"rouge2"`) where: {n} is the n-gram based scoring,
        `"rougeL"`: Longest common subsequence based scoring.
        `"rougeLSum"`: rougeLsum splits text using `"\n"`.
        See details in https://github.com/huggingface/datasets/issues/617
    use_stemmer: Bool indicating whether Porter stemmer should be used to strip word suffixes.
    use_aggregator: Return aggregates if this is set to True
Returns:
    rouge1: rouge_1 (f1),
    rouge2: rouge_2 (f1),
    rougeL: rouge_l (f1),
    rougeLsum: rouge_lsum (f1)
Examples:

    >>> rouge = evaluate.load('rouge')
    >>> predictions = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> results = rouge.compute(predictions=predictions, references=references)
    >>> print(results)
    {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}
"""


class Tokenizer:
    """Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Rouge(evaluate.Metric):

    def __init__(self, allow_aggregate: bool = True, workers: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.verbose = kwargs.get("verbose", True)
        self.allow_aggregate = allow_aggregate
        self.workers = workers

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence")),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/google-research/google-research/tree/master/rouge"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/ROUGE_(metric)",
                "https://github.com/google-research/google-research/tree/master/rouge",
            ],
        )

    def _compute(
        self, predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=True, tokenizer=None,
            split_summaries=False
    ):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        multi_ref = isinstance(references[0], list)

        if tokenizer is not None:
            tokenizer = Tokenizer(tokenizer)

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer,
                                          split_summaries=split_summaries)
        if use_aggregator and self.allow_aggregate:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []

        for ref, pred in tqdm(zip(references, predictions), desc="Computing scores", total=len(references), disable=not self.verbose):
            if multi_ref:
                score = scorer.score_multi(ref, pred)
            else:
                score = scorer.score(ref, pred)
            if use_aggregator and self.allow_aggregate:
                aggregator.add_scores(score)
            else:
                scores.append(score)

        if use_aggregator and self.allow_aggregate:
            result = {}
            # convert the named tuple to a dict

            for k, ag_score in aggregator.aggregate().items():
                result[k + "_low_precision"] = float(ag_score.low.precision)
                result[k + "_low_recall"] = float(ag_score.low.recall)
                result[k + "_low_fmeasure"] = float(ag_score.low.fmeasure)

                result[k + "_mid_precision"] = float(ag_score.mid.precision)
                result[k + "_mid_recall"] = float(ag_score.mid.recall)
                result[k + "_mid_fmeasure"] = float(ag_score.mid.fmeasure)

                result[k + "_high_precision"] = float(ag_score.high.precision)
                result[k + "_high_recall"] = float(ag_score.high.recall)
                result[k + "_high_fmeasure"] = float(ag_score.high.fmeasure)

        else:
            result = {}
            for key in scores[0]:
                result[key + "_precision"] = []
                result[key + "_recall"] = []
                result[key + "_fmeasure"] = []
                for score in scores:
                    s = score[key]
                    result[key + "_precision"].append(s.precision)
                    result[key + "_recall"].append(s.recall)
                    result[key + "_fmeasure"].append(s.fmeasure)

        return result