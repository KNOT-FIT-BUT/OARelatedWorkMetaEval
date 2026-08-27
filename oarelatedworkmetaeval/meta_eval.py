import json
import logging
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from classconfig import ConfigurableMixin, CreatableMixin, ConfigurableValue, \
    ListOfConfigurableSubclassFactoryAttributes, RelativePathTransformer, ConfigurableSubclassFactory
from classconfig.validators import IntegerValidator, AnyValidator, IsNoneValidator, MinValueIntegerValidator, \
    ValueInIntervalFloatValidator, BoolValidator
from datasets import load_from_disk, load_dataset
from matplotlib.patches import Patch
from scipy.stats import shapiro, kendalltau, bootstrap
from scipy.stats._resampling import BootstrapResult
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm

from oarelatedworkmetaeval.metrics import Metric


def tau_wrapper(human_sample, metric_sample):
    return kendalltau(human_sample, metric_sample).statistic


class MetaEvalCoarseWorkflow(ConfigurableMixin, CreatableMixin):
    """
    Workflow for meta evaluation of evaluators on the coarse (whole related work section) level.
    """

    dataset_path: Optional[str] = ConfigurableValue(desc="Dataset path or None to load it from HF hub",
                                                    user_default=None,
                                                    transform=RelativePathTransformer(allow_none=True))

    metrics: list[Metric] = ListOfConfigurableSubclassFactoryAttributes(
        configurable_subclass_factory=ConfigurableSubclassFactory(Metric),
        desc="List of metrics to evaluate",
    )
    results_file: str = ConfigurableValue(
        desc="Path to save the scores and correlation for each sample",
        user_default="coarse_results.jsonl",
        transform=RelativePathTransformer()
    )
    violin_plot_file: str = ConfigurableValue(
        desc="Path to save the violin plot of correlations",
        user_default="coarse_violin_plot.pdf",
        transform=RelativePathTransformer()
    )
    bootstrap_resamples: int = ConfigurableValue(
        desc="Bootstrap number of bootstrap samples",
        user_default=10000,
        validator=MinValueIntegerValidator(1)
    )
    confidence_level: float = ConfigurableValue(
        desc="Confidence level for bootstrap confidence interval",
        user_default=0.95,
        validator=ValueInIntervalFloatValidator(0, 1, left_inclusive=False, right_inclusive=False)
    )
    random_seed: int | None = ConfigurableValue(
        desc="Random seed",
        validator=AnyValidator([IntegerValidator(), IsNoneValidator()])
    )
    reference_based_color: str = ConfigurableValue(
        desc="Reference based color for violin plot",
        user_default="#ff7f0e"
    )
    reference_free_color: str = ConfigurableValue(
        desc="Reference free color for violin plot",
        user_default="#1f77b4"
    )
    go_color: str = ConfigurableValue(
        desc="Greedy oracle color for violin plot",
        user_default="#2ca02c",
        voluntary=True
    )
    violin_width: float = ConfigurableValue(
        desc="Width of each violin in the plot. Smaller values produce thinner distributions.",
        user_default=0.5,
        voluntary=True,
        validator=ValueInIntervalFloatValidator(0, 1, left_inclusive=False, right_inclusive=True)
    )
    violin_row_spacing: float = ConfigurableValue(
        desc="Vertical distance between adjacent violin rows in data units. "
             "Should be >= violin_width to avoid overlap. Lower values bring rows closer together.",
        user_default=1.0,
        voluntary=True,
        validator=ValueInIntervalFloatValidator(0, 10, left_inclusive=False, right_inclusive=True)
    )

    def __post_init__(self):
        if self.dataset_path and (dataset_path := Path(self.dataset_path)).exists():
            self.dataset = load_from_disk(dataset_path)
        else:
            self.dataset = load_dataset(
                "BUT-FIT/OARelatedWorkMetaEval",
                "duels",
                split="train"
            )

    def load_human_eval_data(self) -> dict[str, dict[str, float]]:
        """
        Read the human evaluation data with factuality rates.

        :return: A dictionary mapping paper IDs to a dictionary of model names and their corresponding factuality rates.
        """

        self.human_eval_data = {}
        for item in self.dataset:
            if item["target_paper_id"] not in self.human_eval_data:
                self.human_eval_data[item["target_paper_id"]] = {}
            for i, m in enumerate(item["model_order"]):
                if m in self.human_eval_data[item["target_paper_id"]]:
                    # first occurence contains statements
                    assert item[
                               f"rw{i + 1}_factuality"] is None, f"Expected factuality rate to be None for repeated model {m} in paper {item['target_paper_id']}, but got {item[f'rw{i + 1}_factuality']}"
                    continue
                if item[f"rw{i + 1}_factuality"] is not None:
                    # there are two rw sections without statements
                    self.human_eval_data[item["target_paper_id"]][m] = item[f"rw{i + 1}_factuality"]

        logging.info(f"Loaded human evaluation data for {len(self.human_eval_data)} papers.")

    def load_summaries(self) -> dict[str, dict[str, str] | str | None]:
        """
        Loads generated summaries, used inputs and human written reference.

        :return: A dictionary mapping paper IDs to a dictionary of model names and their corresponding summaries.
            [target_paper_id]
                "target_paper": str,  # markdown format
                "cited_papers": str,  # markdown format
                "reference": str,  # markdown format
                "models": {
                    [model_name]: str  # markdown format
                }
            }
        """
        self.summaries_data = {}

        for item in self.dataset:
            if item["target_paper_id"] not in self.summaries_data:
                self.summaries_data[item["target_paper_id"]] = {
                    "target_paper": item["txt_target_paper"],
                    "cited_papers": item["txt_cited_papers"],
                    "reference": None,
                    "models": {}
                }

            for i, model in enumerate(item["model_order"]):
                if item[f"rw{i + 1}_factuality"] is not None:  # only interested in statements with assest factuality
                    if model not in self.summaries_data[item["target_paper_id"]]["models"]:
                        self.summaries_data[item["target_paper_id"]]["models"][model] = item[f"txt_rw_{i + 1}"]

                if model == "human" and self.summaries_data[item["target_paper_id"]]["reference"] is None:
                    self.summaries_data[item["target_paper_id"]]["reference"] = item[f"txt_rw_{i + 1}"]

        number_of_summaries = sum(len(paper_data["models"]) for paper_data in self.summaries_data.values())
        logging.info(f"Loaded {number_of_summaries} summaries for {len(self.summaries_data)} papers.")

    def eval_samples(self, metric: Metric) -> list[float]:
        """
        Evaluates samples with given metric.

        :param metric: metric to evaluate
        :return: list of metric scores for all samples
        """

        assert hasattr(self,
                       "summaries_data") and self.summaries_data is not None, "Summaries data not loaded. Please run load_summaries() first."

        all_metric_scores = []

        # evaluate samples with given metric
        for target_paper_id, target_paper_data in self.summaries_data.items():
            for model_name, model_summary in target_paper_data["models"].items():
                if metric.skip_humans is True and model_name.lower() == "human":
                    continue

                if metric.reference_free is not True and model_name.lower() == "human":
                    continue

                metric_arguments = {
                    "target_paper_id": target_paper_id,
                    "gen_model": model_name,
                    "summary": model_summary,
                    "reference": target_paper_data["reference"],
                    "target_paper": target_paper_data["target_paper"],
                    "cited_papers": target_paper_data["cited_papers"]
                }

                metric_score = metric(**metric_arguments)

                all_metric_scores.append(metric_score)

        return all_metric_scores

    def get_human_scores_for_summaries(self) -> tuple[list[float], list[float]]:
        """
        Gets human evaluation scores for all summaries.

        :return: list of human evaluation scores for all summaries
            list of human evaluation scores for reference based evaluation (human summaries are omitted)
        """

        assert hasattr(self, "summaries_data") and self.summaries_data is not None, \
            "Summaries data not loaded. Please run load_summaries() first."
        assert hasattr(self, "human_eval_data") and self.human_eval_data is not None, \
            "Human eval data not loaded. Please run load_human_eval_data() first."

        human_scores: list[float] = []
        human_scores_reference_based: list[float] = []

        # Keep the same traversal order as eval_samples() for correlation alignment.
        for target_paper_id, target_paper_data in self.summaries_data.items():
            for model_name in target_paper_data["models"]:
                if target_paper_id not in self.human_eval_data or model_name not in self.human_eval_data[
                    target_paper_id]:
                    raise KeyError(
                        f"Missing human score for model {model_name} in paper {target_paper_id}."
                    )
                human_scores.append(self.human_eval_data[target_paper_id][model_name])
                if model_name.lower() != "human":
                    human_scores_reference_based.append(self.human_eval_data[target_paper_id][model_name])

        return human_scores, human_scores_reference_based

    def plot_violin(self, metrics: Sequence[Metric], bootstrap_res: Sequence[BootstrapResult]):
        """
        Plots results for metric. It will create one figure with multiple violin plots for each metric.
        It will visualize the distribution and confidence interval for each metric.

        On the x axis there will be the correlation and at the y axis the metric.
        It distinguishes between reference free and reference based version of given metric with different colors.

        :param metrics: list of metrics to plot
        :param bootstrap_res: list of bootstrap result containing the distribution and confidence interval to plot
            it corresponds to metric in metrics sequence.
        """
        if len(metrics) == 0 or len(bootstrap_res) == 0:
            raise ValueError("Both metrics and bootstrap results must be non-empty.")
        if len(metrics) != len(bootstrap_res):
            raise ValueError(
                f"Expected same number of metrics and bootstrap results, got {len(metrics)} and {len(bootstrap_res)}."
            )

        positions: list[int] = []
        distributions: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[str] = []
        ci_segments: list[tuple[int, float, float]] = []

        # make's sure that the metric with the same name ar on the same position (reference free/reference based on the same position)
        name_position = {}

        for i, (metric, boot) in enumerate(zip(metrics, bootstrap_res), start=1):
            distribution = np.asarray(boot.bootstrap_distribution, dtype=float)
            distribution = distribution[np.isfinite(distribution)]
            if distribution.size == 0:
                raise ValueError(
                    f"Bootstrap distribution for metric {metric.name} is empty after filtering NaN/Inf values.")

            ci_low = float(boot.confidence_interval.low)
            ci_high = float(boot.confidence_interval.high)
            if not np.isfinite(ci_low) or not np.isfinite(ci_high):
                raise ValueError(f"Invalid confidence interval for metric {metric.name}: ({ci_low}, {ci_high}).")
            if ci_low > ci_high:
                ci_low, ci_high = ci_high, ci_low

            if metric.name not in name_position:
                name_position[metric.name] = (len(metrics) - (len(name_position) + 1)) * self.violin_row_spacing

            positions.append(name_position[metric.name])

            distributions.append(distribution)
            labels.append(f"{metric.name}")

            color = {
                True: self.reference_free_color,
                False: self.reference_based_color,
                "go": self.go_color
            }[metric.reference_free]

            colors.append(color)
            ci_segments.append((name_position[metric.name], ci_low, ci_high, color))

        fig_height = self.violin_width * max(positions) + 1.0
        fig, ax = plt.subplots(figsize=(10, fig_height))

        violin = ax.violinplot(
            distributions,
            positions=positions,
            vert=False,
            showmeans=False,
            showmedians=True,
            widths=self.violin_width,
        )

        # Apply per-violin colors from `colors`.
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.35)

        # Color extrema lines (the long center line + min/max caps) per violin.
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in violin:
                violin[key].set_color(colors)  # one color per distribution
                violin[key].set_linewidth(2.0 if key == "cmedians" else 1.0)

        for pos, ci_low, ci_high, color in ci_segments:
            ax.hlines(y=pos, xmin=ci_low, xmax=ci_high, color=color, linewidth=6, alpha=0.6)

        label_fontsize = 18
        title_fontsize = 22

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=label_fontsize)
        ax.set_xlabel("Kendall tau correlation", fontsize=label_fontsize)
        ax.set_title("OARelatedWork Meta-Evaluation", fontsize=title_fontsize)
        ax.tick_params(axis="x", labelsize=label_fontsize)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        y_pad = self.violin_width / 2 + 0.05
        ax.set_ylim(min(positions) - y_pad, max(positions) + y_pad)

        legend_handles = []
        if any(m.reference_free is True for m in metrics):
            legend_handles.append(
                Patch(
                    facecolor=self.reference_free_color,
                    edgecolor=self.reference_free_color,
                    alpha=0.35,
                    label="Ref. free",
                )
            )
        if any(m.reference_free is False for m in metrics):
            legend_handles.append(
                Patch(
                    facecolor=self.reference_based_color,
                    edgecolor=self.reference_based_color,
                    alpha=0.35,
                    label="Ref. based",
                )
            )

        if any(m.reference_free == "go" for m in metrics):
            legend_handles.append(
                Patch(
                    facecolor=self.go_color,
                    edgecolor=self.go_color,
                    alpha=0.35,
                    label="Greedy oracle based",
                )
            )

        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left", fontsize=label_fontsize)

        output_path = Path(self.violin_plot_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def __call__(self):
        self.load_human_eval_data()
        self.load_summaries()

        human_scores, human_scores_reference_based = self.get_human_scores_for_summaries()
        res = []
        with open(self.results_file, "w") as f, tqdm(total=len(self.metrics), desc="Evaluating metrics") as pbar:
            for i, metric in enumerate(self.metrics):
                pbar.set_description(f"Evaluating metric {metric.name}")
                use_scores = human_scores

                if metric.reference_free is not True or metric.skip_humans:
                    use_scores = human_scores_reference_based

                try:
                    scores = self.eval_samples(metric)
                    shapiro_test = shapiro(np.array(scores))
                    correlation = kendalltau(use_scores, scores)

                    bootstrap_res = bootstrap(
                        data=(use_scores, scores),
                        statistic=tau_wrapper,
                        paired=True,
                        vectorized=False,
                        n_resamples=self.bootstrap_resamples,
                        random_state=self.random_seed,
                        confidence_level=self.confidence_level,
                    )
                    res.append(bootstrap_res)
                    print(json.dumps({
                        "metric_index": i,
                        "metric": metric.name,
                        "reference_free": metric.reference_free,
                        "shapiro_test": {
                            "statistic": shapiro_test.statistic,
                            "p_value": shapiro_test.pvalue
                        },
                        "correlation": {
                            "statistic": correlation.statistic,
                            "p_value": correlation.pvalue
                        },
                        "bootstrap": {
                            "confidence_interval_low": bootstrap_res.confidence_interval.low,
                            "confidence_interval_high": bootstrap_res.confidence_interval.high,
                            "distribution": bootstrap_res.bootstrap_distribution.tolist()
                        }
                    }, ensure_ascii=False), file=f)

                    self.plot_violin(
                        metrics=self.metrics[:len(res)],
                        bootstrap_res=res
                    )
                    pbar.update(1)
                except Exception as e:
                    print(f"Error evaluating metric {i}:{metric.name}: {e}")
                    print(traceback.format_exc())
                    continue


class StatementClassificationResultFile(ConfigurableMixin):
    name: str = ConfigurableValue(desc="Name of classification model")
    path: str = ConfigurableValue(desc="Path to results file", transform=RelativePathTransformer())
    reference_free: bool = ConfigurableValue(
        desc="Whether the results are for reference free or reference based setup",
        validator=BoolValidator(),
    )

    def __post_init__(self):
        self.results = {}
        with open(self.path, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.results[int(record["id"])] = record["classification"]

    def __call__(self, statement_id: str) -> str:
        if statement_id not in self.results:
            raise KeyError(f"Statement ID {statement_id} not found in results file {self.path}.")
        return self.results[statement_id]


class MetaEvalStatementWorkflow(ConfigurableMixin, CreatableMixin):
    """
    Workflow for meta evaluation of evaluators on the statement level.
    """

    dataset_path: Optional[str] = ConfigurableValue(desc="Dataset path or None to load it from HF hub",
                                                    user_default=None,
                                                    transform=RelativePathTransformer(allow_none=True))

    results: list[StatementClassificationResultFile] = ListOfConfigurableSubclassFactoryAttributes(
        configurable_subclass_factory=ConfigurableSubclassFactory(StatementClassificationResultFile),
        desc="Files with statement classification results in jsonl format with `id` and `classification` fields.",
    )

    bootstrap_resamples: int = ConfigurableValue(
        desc="Bootstrap number of bootstrap samples",
        user_default=10000,
        validator=MinValueIntegerValidator(1)
    )

    confidence_level: float = ConfigurableValue(
        desc="Confidence level for bootstrap confidence interval",
        user_default=0.95,
        validator=ValueInIntervalFloatValidator(0, 1, left_inclusive=False, right_inclusive=False)
    )

    random_seed: int | None = ConfigurableValue(
        desc="Random seed",
        validator=AnyValidator([IntegerValidator(), IsNoneValidator()])
    )

    def __post_init__(self):
        dataset_path = Path(self.dataset_path)
        if dataset_path.exists():
            self.dataset = load_from_disk(dataset_path)
        else:
            self.dataset = load_dataset(
                "BUT-FIT/OARelatedWorkMetaEval",
                "statements",
                split="train"
            )

        self.eval = evaluate.load("mdocekal/precision_recall_fscore_accuracy", average=None)

    def load_human_eval_data(self) -> dict[str, dict[str, float]]:
        """
        Read the human evaluation data with factuality rates.

        :return: A dictionary mapping paper IDs to a dictionary of model names and their corresponding factuality rates.
        """

        self.human_eval_data = {}
        self.flags = {}
        self.reference_based_flag = {}  #
        self.associated_model = {}
        for item in self.dataset:
            label = json.loads(item["label"])
            self.human_eval_data[item["id"]] = label["meta"]
            self.flags[item["id"]] = {"All"}
            if item["differ"]:
                self.flags[item["id"]].add("Adversarial Subset")
            else:
                self.flags[item["id"]].add("Random Subset")

            if label["82"] != label["83"]:
                self.flags[item["id"]].add("Human Disagreement Subset")

            self.reference_based_flag[item["id"]] = "human" not in item["statement_id"]

            self.associated_model[item["id"]] = item["statement_id"].split("_", maxsplit=1)[1].rsplit("_", maxsplit=1)[
                0]

        logging.info(f"Loaded human evaluation data for {len(self.human_eval_data)} statements.")

    def print_res(self, name: str, references: list[str], predictions: list[str], labels: list[str]):
        """
        Prints evaluation report.

        :param name: Name of the evaluation report.
        :param references: ground truth references.
        :param predictions: predictions.
        :param labels: ground truth labels.
        """

        print(f"{name}:")
        report = classification_report(references, predictions, labels=labels)
        print(report)

        kappa = cohen_kappa_score(references, predictions)
        print(f"\tCohen's kappa: {kappa}")

        print("\tConfusion matrix:")
        cm = confusion_matrix(references, predictions, labels=labels)

        print("\trows=true labels, cols=predicted labels")
        print("\t" + " | ".join(["{:>28}".format("")] + [f"{lbl:>28}" for lbl in labels]))
        for i, row_label in enumerate(labels):
            row_vals = " | ".join(f"{v:>28}" for v in cm[i])
            print(f"\t{row_label:>28} | {row_vals}")
        print()

    def __call__(self):
        self.load_human_eval_data()
        labels = ["True", "True, but wrong citation", "Unverifiable", "False"]
        issues = {
            "True but wrong citation": "True, but wrong citation",
        }

        for res in self.results:
            # let's check the true rate per model generating summary

            true_rates = defaultdict(int)
            true_rates_cnts = defaultdict(int)
            for k, v in self.human_eval_data.items():
                if res(k) == "True":
                    true_rates[self.associated_model[k]] += 1
                true_rates_cnts[self.associated_model[k]] += 1

            print(f"True rates for {res.name}:")
            for model in sorted(true_rates.keys()):
                print(
                    f"\t{model}: {true_rates[model]}/{true_rates_cnts[model]} = {true_rates[model] / true_rates_cnts[model]:.2f}")

            for statement_type in ["All", "Random Subset", "Adversarial Subset", "Human Disagreement Subset"]:
                predictions, references = [], []
                for k, v in self.human_eval_data.items():
                    if statement_type not in self.flags[k]:
                        continue
                    if not res.reference_free and not self.reference_based_flag[k]:
                        continue

                    # known issues fix

                    if res(k) in issues:
                        predictions.append(issues[res(k)])
                    else:
                        predictions.append(res(k))
                    references.append(v)

                self.print_res(
                    name=f"{statement_type} " + res.name,
                    references=references,
                    predictions=predictions,
                    labels=labels,
                )
                strict_dict = {
                    "True": "Ok",
                    "True, but wrong citation": "Faulty",
                    "Unverifiable": "Faulty",
                    "False": "Faulty",
                }
                self.print_res(
                    name=f"{statement_type} " + res.name + " (strict)",
                    references=[strict_dict[r] for r in references],
                    predictions=[strict_dict[p] for p in predictions],
                    labels=["Ok", "Faulty"]
                )

                merge_dict = {
                    "True": "True",
                    "True, but wrong citation": "True",
                    "Unverifiable": "False",
                    "False": "False",
                }
                self.print_res(
                    name=f"{statement_type} " + res.name + " (merge)",
                    references=[merge_dict[r] for r in references],
                    predictions=[merge_dict[p] for p in predictions],
                    labels=["True", "False"]
                )

                """
    
                bootstrap_precision = bootstrap(
                    data=(predictions, references),
                    statistic=lambda pred, ref: self.eval.compute(predictions=pred, references=ref, average="macro")["precision"],
                    paired=True,
                    vectorized=False,
                    n_resamples=self.bootstrap_resamples,
                    random_state=self.random_seed,
                    confidence_level=self.confidence_level,
                )
                bootstrap_recall = bootstrap(
                    data=(predictions, references),
                    statistic=lambda pred, ref: self.eval.compute(predictions=pred, references=ref, average="macro")[
                        "recall"],
                    paired=True,
                    vectorized=False,
                    n_resamples=self.bootstrap_resamples,
                    random_state=self.random_seed,
                    confidence_level=self.confidence_level,
                )
                bootstrap_f1 = bootstrap(
                    data=(predictions, references),
                    statistic=lambda pred, ref: self.eval.compute(predictions=pred, references=ref, average="macro")[
                        "f1"],
                    paired=True,
                    vectorized=False,
                    n_resamples=self.bootstrap_resamples,
                    random_state=self.random_seed,
                    confidence_level=self.confidence_level,
                )
                """
