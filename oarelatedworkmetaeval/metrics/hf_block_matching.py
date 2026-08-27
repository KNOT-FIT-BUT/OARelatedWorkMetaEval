# -*- coding: UTF-8 -*-
"""
Created on 15.09.23

:author:     Martin Dočekal
"""
import math
from functools import lru_cache
from typing import Sequence, Dict, Iterable, List, Tuple, Optional

import datasets
import evaluate
import munkres
import numpy as np
from evaluate import Metric
from tqdm import tqdm
from whitespacetokenizer import whitespace_tokenizer

_DESCRIPTION = """
Meta metric for calculating similarity of blocks.

It is comparing best matching blocks using given metric.

It is called meta metric because it is using other metrics operating on plain string level, 
but upgrades it to block level.
"""

_KWARGS_DESCRIPTION = """
Produces similarity of multiple text blocks.

Args:
    predictions: Predicted blocks. 
    references: References in form of blocks. 
    Returns:
        block similarity
Examples:
    >>> metric = BlockMatchingMetric(evaluate.load('rouge'), "rouge2")
    >>> metric.compute(references=[
            ["text of block 0", "text of block 1", "text of block 2"]
        ], predictions=[
            ["text of block 0", "text of block 1"]
        ])
    {'recall': 0.6666666666666666, 'precision': 1.0, 'f1': 0.8}

"""


class BlockMatchingMetric(evaluate.Metric):
    """
    Meta metric for calculating similarity of blocks.

    It is comparing best matching blocks using given metric.

    It is called meta metric because it is using other metrics operating on plain string level,
    but upgrades it to block level.
    """

    def __init__(self, metric: evaluate.Metric, sel_key: str, row_col_norm: bool = False, verbose: bool = True, **kwargs):
        """
        Initializes metric.

        :param metric: metric that should be used for string similarity
            It should return value in range [0,1] where 0 means no similarity and 1 means perfect similarity.
        :param sel_key: Key that will be used to select value from dictionary returned by metric.
        :param row_col_norm: If True then the final similarity will be normalized by sum of similarities in its row in
            case of recall and by sum of similarities in its column in case of precision.
        :param verbose: If True then progress bar is shown.
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.sel_key = sel_key
        self.row_col_norm = row_col_norm
        self.verbose = verbose

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation="",
            features=datasets.Features({
                "predictions": datasets.Sequence(datasets.Value("string")),
                "references": datasets.Sequence(datasets.Value("string"))
            }),
        )

    @lru_cache(maxsize=1000)
    def cached_text_similarity(self, text1: str, text2: str) -> float:
        """
        Computes similarity of two texts.

        :param text1: First text.
            act as a reference.
        :param text2: Second text.
            act as a prediction.
        :return: Similarity of two texts.
        """
        return self.metric.compute(
            references=[text1],
            predictions=[text2]
        )[self.sel_key]

    def calc_similarities(self, x_blocks: Sequence[str], y_blocks: Sequence[str]) -> List[List[float]]:
        """
        Computes similarity matrix of two sets of blocks.

        :param x_blocks: First set of blocks.
            act as a reference.
        :param y_blocks: Second set of blocks.
            act as a prediction.
        :return: Similarity matrix.
        """
        references = [x for x in x_blocks for _ in y_blocks]
        predictions = [y for _ in x_blocks for y in y_blocks]

        all_similarities = self.metric.compute(
            references=references,
            predictions=predictions
        )[self.sel_key]

        return [
            [all_similarities[i * len(y_blocks) + j] for j in range(len(y_blocks))]
            for i in range(len(x_blocks))
        ]

    def match_blocks(self, ref_blocks: Sequence[str], pred_blocks: Sequence[str]) -> Tuple[float, float, float]:
        """
        Computes similarity of blocks.

        :param ref_blocks: Reference blocks.
        :param pred_blocks: Predicted blocks.
        :return: Tuple of recall, precision and f1 score.
        """

        similarities = self.calc_similarities(ref_blocks, pred_blocks)  # len(ref_blocks) x len(pred_blocks) matrix

        # will use Hungarian algorithm to find best matching blocks
        cost_matrix = munkres.make_cost_matrix(similarities)

        m = munkres.Munkres()
        indexes = m.compute(cost_matrix)

        if self.row_col_norm:
            np_sim = np.array(similarities)
            abs_val_sim = np.abs(np_sim)
            row_sum = abs_val_sim.sum(axis=1, keepdims=True) + 1e-10
            col_sum = abs_val_sim.sum(axis=0, keepdims=True) + 1e-10
            row_norm = np_sim / row_sum
            col_norm = np_sim / col_sum

            recall = 0
            precision = 0

            for i, j in indexes:
                recall += row_norm[i][j]
                precision += col_norm[i][j]

            recall /= len(ref_blocks)
            precision /= len(pred_blocks)
        else:
            score = sum(similarities[i][j] for i, j in indexes)

            recall = score / len(ref_blocks)
            precision = score / len(pred_blocks)

        f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

        return recall, precision, f1

    def _compute(self, references: Sequence[Sequence[str]], predictions: Sequence[Sequence[str]]) -> Dict[str, float]:
        """
        Computes blocks similarity.

        :param references: Reference blocks.
        :param predictions: Predicted blocks.
        :return: block similarity.
        """

        # get average recall, precision and f1 score

        recall = []
        precision = []
        f1 = []

        for ref, pred in tqdm(zip(references, predictions), disable=not self.verbose, total=len(references),
                              desc="Computing BlockMatchingMetric"):
            r, p, f = self.match_blocks(ref, pred)
            recall.append(r)
            precision.append(p)
            f1.append(f)

        return {
            "recall": sum(recall) / len(recall),
            "precision": sum(precision) / len(precision),
            "f1": sum(f1) / len(f1)
        }


class AParBlockMatchingMetric(BlockMatchingMetric):

    def __init__(self, metric: Metric, sel_key: str, row_col_norm: bool = False, aggr_multi_ref="max",
                 artificial_paragraphs: int = 0, sim_type="f1", verbose: bool = True):
        """

        Args:
            metric: metric that should be used for string similarity
                It should return value in range [0,1] where 0 means no similarity and 1 means perfect similarity.
            sel_key: Key that will be used to select value from dictionary returned by metric.
            row_col_norm: If True then the final similarity will be normalized by sum of similarities in its row in
                case of recall and by sum of similarities in its column in case of precision.
            aggr_multi_ref: How to aggregate multiple references.
                Possible values are:
                    - "max" - maximum similarity of all references
                    - "mean" - mean similarity of all references

            artificial_paragraphs: values > 0 means that we will create artificial paragraphs using
                given number of sentences
                NOT SUPPORTED ANYMORE
            sim_type: Type of block matching similarity. WARNING it is not the same as ROUGE2 type, it is always f1.
                Choose from:
                    - "f1" - f1 score
                    - "recall" - recall
                    - "precision" - precision
            verbose: If True then progress bar is shown.
        """
        super().__init__(metric, sel_key, row_col_norm, verbose)
        self.aggr_multi_ref = aggr_multi_ref
        self.artificial_paragraphs = artificial_paragraphs

        if sim_type not in ["f1", "recall", "precision"]:
            raise ValueError(f"Unknown type {sim_type}.")

        self.sim_type = sim_type

    @staticmethod
    def make_blocks(sentences: Iterable[Sequence[str]], artificial_paragraphs: int = 0,
                    sentence_sep: str = " ") -> List[List[str]]:
        """
        Converts sentences to blocks


        Args:
            sentences: it expects iterable of iterables because it is expected to be used in batch fashion.
             Thus, each element is an iterable with sentences that should be converted to tree.
            artificial_paragraphs: values > 0 means that we will create artificial paragraphs using
                given number of sentences
            sentence_sep: separator of sentences
        Returns:
            blocks
        """
        res = []

        for sample_sentences in sentences:
            paragraphs = []
            for from_sen in range(0, len(sample_sentences), artificial_paragraphs):
                to_sen = min(from_sen + artificial_paragraphs, len(sample_sentences))
                paragraphs.append(sentence_sep.join(sample_sentences[from_sen:to_sen]))
            res.append(paragraphs)

        return res

    def block_splitter(self, text: str, split_length: int) -> list[str]:
        """
        Splits the text into blocks of max split_length character length.

        The splitting is word aware, meaning that the text will be split on a whitespace.

        :param text: Text to be split.
        :param split_length: Max length of each block.
        :return: split blocks
        """

        blocks = []
        if not text:
            return blocks

        tokens = whitespace_tokenizer(text)
        start_token = tokens[0]
        for i, t in enumerate(tokens):
            if t[2] - start_token[1] > split_length:
                blocks.append(text[start_token[1]:tokens[i - 1][2]])
                start_token = t
            if len(blocks) == math.ceil(len(text) / split_length) - 1:
                # last block
                blocks.append(text[t[1]:])
                break

        return blocks

    def convert_to_blocks(self, text: str, split_length: Optional[int] = None) -> List[str]:
        """
        It will simply split the text by new line. Every line will be stripped and empty lines will be removed.

        :param text: Text to be converted to blocks.
        :param split_length: This attribute activates block splitting.
        If set, the text will be split into blocks of max split_length character length
        :return: New result with converted fields.
        """

        text = text.split("\n")

        # strip and remove empty lines
        text = [x.strip() for x in text if x.strip()]

        # split into blocks
        if split_length is not None:
            text = [block for line in text for block in self.block_splitter(line, split_length)]

        return text

    def score(self, src: str, tgts: Sequence[str]) -> float:
        """
        Computes similarity of source and target sequence.

        Args:
            src: Source sequence.
            tgts: References of target sequence.

        Returns:
            Similarity of source and target sequence.
        """
        texts = [src]
        texts.extend(tgts)

        if self.artificial_paragraphs > 0:
            raise NotImplementedError("Artificial paragraphs are not supported anymore.")
        else:
            blocks = [
                self.convert_to_blocks(text, split_length=512) for text in texts
            ]

        src_blocks = blocks[0]

        similarities = []

        for t in blocks[1:]:
            similarities.append(self.compute(
                references=[src_blocks],
                predictions=[t]
            )[self.sim_type])

        if self.aggr_multi_ref == "max":
            return max(similarities)
        elif self.aggr_multi_ref == "mean":
            return sum(similarities) / len(similarities)
        else:
            raise ValueError(f"Unknown aggregation method {self.aggr_multi_ref}.")
