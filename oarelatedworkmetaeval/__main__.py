"""
Created on 16.02.26

:author:     Martin Dočekal
"""

import argparse
import csv
import itertools
import json
import sys
import statistics
import numpy as np
from scipy.stats import bootstrap
from rouge_score import rouge_scorer
import textstat
import spacy
import re

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

from classconfig import Config
from datasets import load_dataset, Dataset, load_from_disk
from sympy import true
from windpyutils.visual.text import print_histogram, print_buckets_histogram

from oarelatedworkmetaeval.load import load_and_prepare_annotations
from oarelatedworkmetaeval.meta_eval import MetaEvalCoarseWorkflow, MetaEvalStatementWorkflow


def trans_preference(preference: str, model_order: list[str]) -> str | None:
    """
    Translate preference label to integer.

    :param preference: Preference label as string ("1. is better", "2. is better", "Same quality")
    :param model_order: List of model names in the order they were presented to annotators (e.g., ["model_a", "model_b"])
    :return: Name of the preferred model or None if "Same quality"
    """
    if preference == "Same quality":
        return None
    elif preference == "1. is better":
        return model_order[0]
    elif preference == "2. is better":
        return model_order[1]
    else:
        raise ValueError(f"Unknown preference label: {preference}")


def convert_to_hf_dataset(annotations: dict, dataset: Dataset, label_studio_input: list, meta_annotations: dict,
                          text_form_for_target_paper: dict, go_text_form_for_target_paper: dict) -> Dataset:
    """
    Convert the annotation results to Hugging Face Dataset format.

    :param annotations: Dictionary containing annotation results loaded with function load_and_prepare_data.
    :param dataset: Original Hugging Face Dataset containing the data.
    :param label_studio_input: List of Label Studio input data containing model order and original generated related work for each task.
    :param meta_annotations: Dictionary containing loaded meta annotations for statements.
    :param text_form_for_target_paper: Text representations of target, cited papers.
    :param go_text_form_for_target_paper: GO Text representations of target, cited papers.
    :return: Hugging Face Dataset with annotations merged in.
    """

    data = []
    annotators = [82, 83]  # annotators identifiers
    id_2_index = {item: idx for idx, item in enumerate(dataset["id"])}

    for target_paper_id, tasks in annotations.items():
        orig_data = dataset[id_2_index[target_paper_id]]

        for task_id in tasks:
            task = tasks[task_id]

            randomly_selected_statements, randomly_selected_differ_statements = {}, {}
            for annotator_id in annotators + ["model"]:
                for store_to, load_from in [
                    (randomly_selected_statements, task["annotations"][annotator_id]["randomly_selected_statements"]),
                    (randomly_selected_differ_statements,
                     task["annotations"][annotator_id]["randomly_selected_differ_statements"])]:
                    for statement_id, statement in load_from.items():
                        if statement_id in store_to:
                            store_to[statement_id]["label"][annotator_id] = str(statement["label"])

                            for attribute in ["edited_start", "edited_end", "rendered_edited_start",
                                              "rendered_edited_end"]:
                                store_to[statement_id][attribute][annotator_id] = statement[attribute]

                            if "evidence" in statement:
                                store_to[statement_id]["evidence"] = statement["evidence"]

                        else:
                            store_to[statement_id] = statement

                            if "evidence" in statement:
                                store_to[statement_id]["evidence"] = statement["evidence"]

                            store_to[statement_id]["label"] = {
                                annotator_id: str(statement["label"])
                            }
                            for attribute in ["edited_start", "edited_end", "rendered_edited_start",
                                              "rendered_edited_end"]:
                                store_to[statement_id][attribute] = {
                                    annotator_id: statement[attribute]
                                }

                        model_name = task["model_order"][int(statement_id.split("_")[1]) - 1]
                        unique_statement_id = str(target_paper_id) + "_" + model_name + "_" + \
                                       statement_id.split("_")[-1]

                        meta_eval = meta_annotations.get(unique_statement_id, None)
                        if meta_eval is None:
                            # this means that all annotations are the same, so we can take any of them as the meta annotation
                            # let's just check it is true
                            labels = set(store_to[statement_id]["label"].values())
                            assert len(
                                labels) == 1, f"All annotations should be the same for statement {statement_id} of paper {target_paper_id}, but got {store_to[statement_id]['label']}"
                            meta_eval = next(iter(labels))
                        else:
                            meta_eval = meta_eval["meta label"]

                        meta_eval = meta_eval.strip().lower()
                        # normalize meta_eval to be "True", "True, but wrong citation", "False", "Unverifiable"
                        if meta_eval == "true":
                            meta_eval = "True"
                        elif meta_eval in ["true, but wrong citation", "true but wrong citation"]:
                            meta_eval = "True, but wrong citation"
                        elif meta_eval == "false":
                            meta_eval = "False"
                        elif meta_eval == "unverifiable":
                            meta_eval = "Unverifiable"
                        else:
                            raise ValueError(
                                f"Unknown meta annotation label: {meta_eval} for statement {statement_id} of paper {target_paper_id}")

                        store_to[statement_id]["label"]["meta"] = meta_eval

            rw1_statements_cnt, rw2_statements_cnt = 0, 0
            rw1_true_statements_cnt, rw2_true_statements_cnt = 0, 0
            for statement_id, statement in itertools.chain(randomly_selected_statements.items(),
                                                           randomly_selected_differ_statements.items()):
                if statement_id.startswith("rw_1"):
                    rw1_statements_cnt += 1
                    if statement["label"]["meta"] == "True":
                        rw1_true_statements_cnt += 1
                elif statement_id.startswith("rw_2"):
                    rw2_statements_cnt += 1
                    if statement["label"]["meta"] == "True":
                        rw2_true_statements_cnt += 1

            data.append({
                "id": len(data),
                "model_order": task["model_order"],
                "rw1_factuality": rw1_true_statements_cnt / rw1_statements_cnt if rw1_statements_cnt > 0 else None,
                "rw2_factuality": rw2_true_statements_cnt / rw2_statements_cnt if rw2_statements_cnt > 0 else None,
                "rw_1": label_studio_input[len(data)]["data"]["orig_rw_1"],
                "rw_2": label_studio_input[len(data)]["data"]["orig_rw_2"],
                "preference": [trans_preference(task["annotations"][annotator_id]["preference"], task["model_order"])
                               for annotator_id in annotators],
                "relevance": [trans_preference(task["annotations"][annotator_id]["relevance"], task["model_order"]) for
                              annotator_id in annotators],
                "faithfulness": [
                    trans_preference(task["annotations"][annotator_id]["faithfulness"], task["model_order"]) for
                    annotator_id in annotators],
                "language": [trans_preference(task["annotations"][annotator_id]["language"], task["model_order"]) for
                             annotator_id in annotators],
                "randomly_selected_statements": json.dumps(randomly_selected_statements),
                "randomly_selected_differ_statements": json.dumps(randomly_selected_differ_statements),
                "target_paper_id": target_paper_id,
                "s2orc_id": orig_data["s2orc_id"],
                "mag_id": orig_data["mag_id"],
                "doi": orig_data["doi"],
                "title": orig_data["title"],
                "abstract": orig_data["abstract"],
                "related_work": orig_data["related_work"],
                "hierarchy": orig_data["hierarchy"],
                "authors": orig_data["authors"],
                "year": orig_data["year"],
                "fields_of_study": orig_data["fields_of_study"],
                "referenced": orig_data["referenced"],
                "bibliography": orig_data["bibliography"],
                "non_plaintext_content": orig_data["non_plaintext_content"],
                "annot_rw_1": task["annot_rw_1"],
                "annot_rw_2": task["annot_rw_2"],
                "annot_target_paper": task["annot_target_paper"],
                "annot_cited_papers": task["annot_cited_papers"],
                "txt_target_paper": text_form_for_target_paper[target_paper_id]["target_paper"],
                "txt_cited_papers": text_form_for_target_paper[target_paper_id]["cited_papers"],
                "go_txt_target_paper": go_text_form_for_target_paper[target_paper_id]["target_paper"],
                "go_txt_cited_papers": go_text_form_for_target_paper[target_paper_id]["cited_papers"],
                "txt_rw_1": text_form_for_target_paper[target_paper_id][task["model_order"][0]],
                "txt_rw_2": text_form_for_target_paper[target_paper_id][task["model_order"][1]],
                "txt_rw_reference": text_form_for_target_paper[target_paper_id]["human"],
            })

    return Dataset.from_list(data)


def convert_duel_to_statement_orientation(dataset: Dataset) -> Dataset:
    """
    Convert duel-oriented dataset to statement-oriented dataset.

    :param dataset: Duel oriented dataset to be converted.
    :return: Dataset with statement-oriented dataset.
    """
    data = []
    cnt = set()
    cnt_per_paper = defaultdict(set)
    for record in dataset:
        randomly_selected_statements = json.loads(record["randomly_selected_statements"])
        randomly_selected_differ_statements = json.loads(record["randomly_selected_differ_statements"])

        for (statement_key, statement_value), differ_flag in itertools.chain(
            zip(randomly_selected_statements.items(), [False] * len(randomly_selected_statements)),
            zip(randomly_selected_differ_statements.items(), [True] * len(randomly_selected_differ_statements))
        ):
            statement_key_parts = statement_key.split("_")
            rw = statement_key_parts[1]
            res = {
                "id": len(data),
                "target_paper_id": record["target_paper_id"],
                "model": record["model_order"][int(rw)-1],
                "statement_id": str(record["target_paper_id"]) + "_" + record["model_order"][int(rw)-1] + "_" + statement_key_parts[-1],
                "differ": differ_flag,
            }
            cnt_per_paper[record["target_paper_id"]].add(record["model_order"][int(rw)-1])
            cnt.add(str(record["target_paper_id"]) + "_" + record["model_order"][int(rw)-1])
            statement_value = {
                k: json.dumps(v) if isinstance(v, dict) else v
                for k, v in statement_value.items()
            }
            res |= statement_value
            # add content
            res |= {
                "s2orc_id": record["s2orc_id"],
                "mag_id": record["mag_id"],
                "doi": record["doi"],
                "title": record["title"],
                "abstract": record["abstract"],
                "related_work": record["related_work"],
                "hierarchy": record["hierarchy"],
                "authors": record["authors"],
                "year": record["year"],
                "fields_of_study": record["fields_of_study"],
                "referenced": record["referenced"],
                "bibliography": record["bibliography"],
                "non_plaintext_content": record["non_plaintext_content"],
                "annot_rw_1": record["annot_rw_1"],
                "annot_rw_2": record["annot_rw_2"],
                "annot_target_paper": record["annot_target_paper"],
                "annot_cited_papers": record["annot_cited_papers"],
                "txt_target_paper": record["txt_target_paper"],
                "txt_cited_papers": record["txt_cited_papers"],
                "go_txt_target_paper": record["go_txt_target_paper"],
                "go_txt_cited_papers": record["go_txt_cited_papers"],
                "txt_rw": record["txt_rw_"+rw],
                "txt_rw_reference": record["txt_rw_reference"],
            }
            data.append(res)

    for paper_id, models in cnt_per_paper.items():
        if len(models) != 3:
            print(f"Paper {paper_id} has only {len(models)} models: {models}")
    return Dataset.from_list(data)


def load_meta_annotations(meta_annotations_path: str) -> dict:
    """
    Load meta annotations from a csv file.

    :param meta_annotations_path: Path to the csv file containing meta annotations.
    :return: Dictionary with meta annotations.
    """
    res = {}
    with open(meta_annotations_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            statement_id = record["target_paper_id"] + "_" + record["rw_from_model"] + "_" + \
                           record["statement_id"].split("_")[-1]
            res[statement_id] = record

    return res


def read_txt_representation(source: str, just_inputs: bool = False) -> dict:
    res = {}
    with open(source, "r") as f:
        for line in f:
            data = json.loads(line)
            res[data["target_paper_id"]] = {
                "target_paper": data["target_paper"],
                "cited_papers": data["cited_papers"],
            }

            if not just_inputs:
                for model in ["primera_go_all", "gpt_4o_mini_go_all", "human"]:
                    res[data["target_paper_id"]][model] = data[model]

    return res


def dataset_creation(args):
    """
    Sub-command for dataset creation.
    """

    dataset = load_dataset("BUT-FIT/OARelatedWork",
                           "oa_related_work",
                           split="test")

    with open(args.label_studio_input_file, "r", encoding="utf-8") as f:
        label_studio_input = json.load(f)

    with open(args.randomly_selected_statements, "r", encoding="utf-8") as f:
        randomly_selected_statements = json.load(f)

    with open(args.randomly_selected_differ_statements, "r", encoding="utf-8") as f:
        randomly_selected_differ_statements = json.load(f)

    model_order = [x["data"]["order"] for x in label_studio_input]

    loaded = load_and_prepare_annotations(
        export_path=args.label_studio_annotations,
        model_order=model_order,
        randomly_selected_statements=randomly_selected_statements,
        randomly_selected_differ_statements=randomly_selected_differ_statements,
        label_studio_input=label_studio_input
    )

    meta_eval = load_meta_annotations(args.meta)
    txt_representations = read_txt_representation(args.text_representations)
    go_txt_representations = read_txt_representation(args.go_text_representations, just_inputs=True)

    # the default config is duel oriented
    converted = convert_to_hf_dataset(loaded, dataset, label_studio_input, meta_eval, txt_representations, go_txt_representations)

    target_dir = Path(args.output_dir) / "duels"
    target_dir.mkdir(parents=True, exist_ok=True)
    converted.save_to_disk(str(target_dir / "train"))
    converted.to_json(str(target_dir / "train.jsonl"), orient="records", lines=True)

    # create per statement config
    statement_oriented = convert_duel_to_statement_orientation(converted)

    target_dir = Path(args.output_dir) / "statements"
    target_dir.mkdir(parents=True, exist_ok=True)
    statement_oriented.save_to_disk(str(target_dir / "train"))
    statement_oriented.to_json(str(target_dir / "train.jsonl"), orient="records", lines=True)


def annot_times(args):
    """
    Sub-command for calculating annotation times.
    """
    with open(args.label_studio_input_file, "r", encoding="utf-8") as f:
        label_studio_input = json.load(f)

    model_order = [x["data"]["order"] for x in label_studio_input]

    times_per_model = {
        "human": [],
        "primera_go_all": [],
        "gpt_4o_mini_go_all": [],
        "all": [],
    }

    with open(args.label_studio_annotations, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    for i, duel in enumerate(annotations):
        for annotation in duel["annotations"]:
            if annotation["completed_by"] not in [82, 83]:
                continue
            
            results_ids = [result["id"] for result in annotation["result"] if "id" in result]
            lead_time = annotation["lead_time"]
            
            times_per_model["all"].append(lead_time)

            for order_i, model in enumerate(model_order[i]):
                if any(f"rw_{order_i + 1}" in result_id for result_id in results_ids):
                    times_per_model[model].append(lead_time)


    print("Annotation times (in seconds):")
    for model, times in times_per_model.items():
        print(f"  {model}: {sum(times) / len(times) if times else 0:.2f} ± {statistics.stdev(times) if times else 0:.2f}")

    print("Atomic claims histogram:")
    for model in sorted(times_per_model):
        print(f"  Model: {model}")
        values = defaultdict(int)
        for t in times_per_model[model]:
            values[t] += 1

        if values:
            print_buckets_histogram(values, buckets=20)
        else:
            print("    No atomic-claim data.")


def duel_win_score(args):
    """
    Sub-command for calculating duel win-score.
    """
    if args.dataset is None:
        print("Error: --dataset argument is required for duel_win_score command.", file=sys.stderr)
        sys.exit(1)

    # Load local JSONL dataset from folder
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset folder '{args.dataset}' does not exist.", file=sys.stderr)
        sys.exit(1)

    datasets = load_from_disk(dataset_path)

    attributes = ["preference", "relevance", "faithfulness", "language"]
    scores = {}  # [attribute][model][wins | duels]
    per_paper_scores = defaultdict(lambda: defaultdict(list)) # [attribute][modelvsmodel][paper_index]

    for item in datasets:
        for m in item["model_order"]:
            for a in attributes:
                if a not in scores:
                    scores[a] = {}
                if m not in scores[a]:
                    scores[a][m] = {
                        "wins": 0, "ties": 0, "looses": 0, "aggregated_score": 0, "duels": 0,
                    }
                scores[a][m]["duels"] += 1

        for a in attributes:
            """
            Annotator 1	Annotator 2	Duel Resolution	Points for Model A	Points for Model B
            Model A	Model A	Strong Win for A	1.0	0.0
            Model A	Tie	Leans A	0.75	0.25
            Model A	Model B	Disagreement (Tie)	0.50	0.50
            Tie	Tie	True Tie	0.50	0.50
            Tie	Model B	Leans B	0.25	0.75
            Model B	Model B	Strong Win for B	0.0	1.0
            """
            aggregated_scores = {m: 0 for m in item["model_order"]}

            for m in item[a]:
                if m is None:
                    for t in item["model_order"]:
                        scores[a][t]["ties"] += 1
                        aggregated_scores[t] += 0.5
                else:
                    scores[a][m]["wins"] += 1
                    aggregated_scores[m] += 1.0

                    loser = [t for t in item["model_order"] if t != m][0]
                    scores[a][loser]["looses"] += 1

            sorted_models = tuple(sorted(item["model_order"]))
            per_paper_scores[a][sorted_models].append(
                aggregated_scores[sorted_models[0]] / len(item[a])
            )
            for m in item["model_order"]:
                scores[a][m]["aggregated_score"] += aggregated_scores[m] / len(item[a])

    for a in attributes:
        print(f"Attribute: {a}")
        for m in scores[a]:
            wins = scores[a][m]["wins"]
            duels = scores[a][m]["duels"]
            ties = scores[a][m]["ties"]
            loses = scores[a][m]["looses"]

            aggregated_score = scores[a][m]["aggregated_score"] / duels if duels > 0 else 0

            total_annotations = duels * 2

            win_score = wins / total_annotations if total_annotations > 0 else 0
            tie_score = ties / total_annotations if total_annotations > 0 else 0
            lose_score = loses / total_annotations if total_annotations > 0 else 0

            print(
                f"  Model: {m}, True Win Rate (Aggregated): {aggregated_score:.4f}, Duels: {duels}, "
                f"Wins: {wins}, Ties: {ties}, Loses: {loses}, "
                f"Raw Win %: {win_score:.4f}, Raw Tie %: {tie_score:.4f}, Raw Lose %: {lose_score:.4f}"
            )

        print(f"  Duel-wise scores for attribute {a}:")
        rng = np.random.default_rng(args.bootstrap_seed)
        for model_pair, scores_list in per_paper_scores[a].items():
            if not scores_list:
                print(f"    Model Pair: {model_pair} (score for {model_pair[0]}), no data")
                continue

            scores_arr = np.asarray(scores_list, dtype=float)
            average_score = float(scores_arr.mean())

            if scores_arr.size > 1:
                ci = bootstrap(
                    (scores_arr,),
                    np.mean,
                    n_resamples=args.bootstrap_iterations,
                    confidence_level=args.bootstrap_confidence,
                    method="percentile",
                    random_state=rng,
                ).confidence_interval
                ci_lo, ci_hi = float(ci.low), float(ci.high)
            else:
                ci_lo, ci_hi = average_score, average_score

            if ci_lo <= 0.5 <= ci_hi:
                winner = "tie"
            elif average_score > 0.5:
                winner = model_pair[0]
            else:
                winner = model_pair[1]

            print(
                f"    Model Pair: {model_pair} (score for {model_pair[0]}), "
                f"Average Score: {average_score:.4f}, "
                f"{int(args.bootstrap_confidence * 100)}% CI: [{ci_lo:.4f}, {ci_hi:.4f}], "
                f"Winner: {winner}, Scores: {scores_list}"
            )




def statement_score(args):
    """
    Sub-command for calculating statement score.
    """

    """
    We calculate followings scores:
    
    A. Strict Grounded Factuality Rate
        This is the ultimate standard. The statement is true and correctly attributed and every annotator agrees on it.
    
    B. Loose Factuality Rate
        Every annotator thinks that the statement is either True or True, but wrong citation. This is a more lenient measure that still considers a statement as factual if it is recognized as such by annotators, even if there is some disagreement about the citation.
    
    """

    if args.dataset is None:
        print("Error: --dataset argument is required for duel_win_score command.", file=sys.stderr)
        sys.exit(1)

    # Load local JSONL dataset from folder
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset folder '{args.dataset}' does not exist.", file=sys.stderr)
        sys.exit(1)

    datasets = load_from_disk(dataset_path)

    scores = {
        "all_statements": {}
    }

    with (open(args.output_verifiable_statements, "w") if args.output_verifiable_statements else nullcontext()) as f:
        for item in datasets:
            models = item["model_order"]
            for m in models:
                if m not in scores["all_statements"]:
                    scores["all_statements"][m] = {
                        "Strict Grounded": 0,
                        "Loose Factual": 0,
                        "Weak Factual": 0,
                        "True, but wrong citation": 0,
                        "False": 0,
                        "Unverifiable": 0,
                        "Unverifiable consensual": 0,
                        "Total": 0,
                        "Non Consensual": 0,
                    }

            for a in ["randomly_selected_statements", "randomly_selected_differ_statements"]:
                if a not in scores:
                    scores[a] = {}
                for m in models:
                    if m not in scores[a]:
                        scores[a][m] = {
                            "Strict Grounded": 0,
                            "Loose Factual": 0,
                            "Weak Factual": 0,
                            "True, but wrong citation": 0,
                            "False": 0,
                            "Unverifiable": 0,
                            "Unverifiable consensual": 0,
                            "Total": 0,
                            "Non Consensual": 0,
                        }
                aggregated_statements = {}  # aggregates annotators decisions

                for statement_label, v in json.loads(item[a]).items():
                    if statement_label not in aggregated_statements:
                        aggregated_statements[statement_label] = {
                            "statement": v["text"],
                            "Strict Grounded": True,
                            "Loose Factual": True,
                            "Weak Factual": True,
                            "True, but wrong citation": True,
                            "False": True,
                            "Unverifiable": False,
                            "Unverifiable consensual": True,
                            "Annotations": set()
                        }
                    for annotator_id, label in v["label"].items():
                        if args.meta and annotator_id != "meta":
                            continue

                        if not args.meta and annotator_id not in ["82", "83"]:
                            continue
                        aggregated_statements[statement_label]["Annotations"].add(label)

                        aggregated_statements[statement_label]["Strict Grounded"] = \
                        aggregated_statements[statement_label][
                            "Strict Grounded"] and (label == "True")
                        aggregated_statements[statement_label]["Loose Factual"] = \
                        aggregated_statements[statement_label][
                            "Loose Factual"] and (label in ["True",
                                                            "True, but wrong citation"])

                        aggregated_statements[statement_label]["Weak Factual"] = aggregated_statements[statement_label][
                                                                                     "Weak Factual"] and (
                                                                                             label in ["True",
                                                                                                       "True, but wrong citation",
                                                                                                       "Unverifiable"])

                        aggregated_statements[statement_label]["True, but wrong citation"] = aggregated_statements[statement_label]["True, but wrong citation"] and (label == "True, but wrong citation")

                        aggregated_statements[statement_label]["False"] = aggregated_statements[statement_label]["False"] and (label == "False")

                        aggregated_statements[statement_label]["Unverifiable"] = aggregated_statements[statement_label][
                                                                                     "Unverifiable"] or (
                                                                                             label == "Unverifiable")
                        aggregated_statements[statement_label]["Unverifiable consensual"] = \
                        aggregated_statements[statement_label][
                            "Unverifiable consensual"] and (label == "Unverifiable")

                for statement_label, v in aggregated_statements.items():
                    if args.output_verifiable_statements:
                        if not v["Unverifiable"]:
                            print(json.dumps({
                                "target_paper_id": item["target_paper_id"],
                                "model": item["model_order"][int(statement_label.split("_")[1]) - 1],
                                "statement_label": statement_label,
                                "statement": v["statement"],
                            }), file=f)
                    m = statement_label.split("_")[1]
                    m = item["model_order"][int(m) - 1]

                    if v["Strict Grounded"]:
                        scores[a][m]["Strict Grounded"] += 1
                        scores["all_statements"][m]["Strict Grounded"] += 1

                    if v["Loose Factual"]:
                        scores[a][m]["Loose Factual"] += 1
                        scores["all_statements"][m]["Loose Factual"] += 1

                    if v["Weak Factual"]:
                        scores[a][m]["Weak Factual"] += 1
                        scores["all_statements"][m]["Weak Factual"] += 1

                    if v["True, but wrong citation"]:
                        scores[a][m]["True, but wrong citation"] += 1
                        scores["all_statements"][m]["True, but wrong citation"] += 1

                    if v["False"]:
                        scores[a][m]["False"] += 1
                        scores["all_statements"][m]["False"] += 1

                    if v["Unverifiable"]:
                        scores[a][m]["Unverifiable"] += 1
                        scores["all_statements"][m]["Unverifiable"] += 1

                    if v["Unverifiable consensual"]:
                        scores[a][m]["Unverifiable consensual"] += 1
                        scores["all_statements"][m]["Unverifiable consensual"] += 1

                    scores[a][m]["Total"] += 1
                    scores["all_statements"][m]["Total"] += 1

                    scores[a][m]["Non Consensual"] += len(v["Annotations"]) > 1
                    scores["all_statements"][m]["Non Consensual"] += len(v["Annotations"]) > 1

    for a in scores:
        print(f"{a}")
        for m in scores[a]:
            strict_grounded_rate = scores[a][m]["Strict Grounded"] / scores[a][m]["Total"] if scores[a][m][
                                                                                                  "Total"] > 0 else 0
            loose_factual_rate = scores[a][m]["Loose Factual"] / scores[a][m]["Total"] if scores[a][m][
                                                                                              "Total"] > 0 else 0

            weak_factual_rate = scores[a][m]["Weak Factual"] / scores[a][m]["Total"] if scores[a][m][
                                                                                            "Total"] > 0 else 0

            non_consensual_rate = scores[a][m]["Non Consensual"] / scores[a][m]["Total"] if scores[a][m][
                                                                                                "Total"] > 0 else 0

            true_but_wrong_citation_rate = scores[a][m]["True, but wrong citation"] / scores[a][m]["Total"] if scores[a][m][
                                                                                                    "Total"] > 0 else 0

            false_rate = scores[a][m]["False"] / scores[a][m]["Total"] if scores[a][m]["Total"] > 0 else 0

            unverifiable_rate = scores[a][m]["Unverifiable"] / scores[a][m]["Total"] if scores[a][m][
                                                                                            "Total"] > 0 else 0

            unverifiable_consensual_rate = scores[a][m]["Unverifiable consensual"] / scores[a][m]["Total"] if \
            scores[a][m][
                "Total"] > 0 else 0

            verifiable = scores[a][m]["Total"] - scores[a][m]["Unverifiable consensual"]
            strict_grounded_rate_verifiable = scores[a][m]["Strict Grounded"] / verifiable if verifiable > 0 else 0

            print(
                f"  Model: {m}, Strict Grounded Factuality Rate: {strict_grounded_rate:.4f}, Loose Factuality Rate: {loose_factual_rate:.4f}, Weak Factuality Rate: {weak_factual_rate:.4f}, True, but wrong citation Rate: {true_but_wrong_citation_rate:.4f}, False Rate: {false_rate:.4f}, Strict Grounded Factuality Rate Verifiable: {strict_grounded_rate_verifiable:.4f}, Total Statements: {scores[a][m]['Total']}, Non Consensual: {scores[a][m]['Non Consensual']}, Non Consensual rate: {non_consensual_rate:.4f}, Unverifiable Rate: {unverifiable_rate:.4f}, Unverifiable Consensual Rate: {unverifiable_consensual_rate:.4f}")


def load_verifiable_statements(path_to_verifiable_statements: str) -> set:
    res = set()
    for line in open(path_to_verifiable_statements, "r"):
        data = json.loads(line)
        res.add(str(data["target_paper_id"]) + "_" + data["model"] + "_" + data["statement"])

    return res


def statement_score_from_model_initial(args):
    """
    Sub-command for calculating statement score from model.
    """

    with open(args.random_statements, "r") as f:
        random_statements = json.load(f)

    with open(args.all_statements, "r") as f:
        all_statements = json.load(f)

    scores = {
        "all_statements": {},
        "randomly_selected_statements": {},
        "randomly_selected_differ_statements": {},
    }

    verifiable_statements = load_verifiable_statements(
        args.verifiable_statements) if args.verifiable_statements else None
    for a, loaded_file in [("randomly_selected_statements", random_statements), ("all_statements", all_statements)]:

        for identifier, statements in loaded_file.items():
            model = identifier.split("_", maxsplit=1)[1]

            if model not in scores[a]:
                scores[a][model] = {
                    "True": 0,
                    "Total": 0,
                }

            for statement in statements:
                if verifiable_statements is not None and (
                        identifier + "_" + statement["statement"]) not in verifiable_statements:
                    continue

                scores[a][model]["True"] += statement["is_true"]
                scores[a][model]["Total"] += 1

                if a == "all_statements" and all(x != statement for x in random_statements[identifier]):
                    if model not in scores["randomly_selected_differ_statements"]:
                        scores["randomly_selected_differ_statements"][model] = {
                            "True": 0,
                            "Total": 0,
                        }
                    scores["randomly_selected_differ_statements"][model]["True"] += statement["is_true"]
                    scores["randomly_selected_differ_statements"][model]["Total"] += 1

    for a in scores:
        print(f"{a}")
        for m in scores[a]:
            score = scores[a][m]["True"] / scores[a][m]["Total"] if scores[a][m]["Total"] > 0 else 0
            print(f"  Model: {m}, True Statement Rate: {score:.4f}, Total Statements: {scores[a][m]['Total']}")


def extract_factuality_rates(args):
    if args.dataset is None:
        print("Error: --dataset argument is required for extract_factuality_rates command.", file=sys.stderr)
        sys.exit(1)

    # Load local JSONL dataset from folder
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset folder '{args.dataset}' does not exist.", file=sys.stderr)
        sys.exit(1)

    datasets = load_from_disk(dataset_path)

    res = {}
    for item in datasets:
        if item["target_paper_id"] not in res:
            res[item["target_paper_id"]] = {}
        for i, m in enumerate(item["model_order"]):
            if m in res[item["target_paper_id"]]:
                # first occurence contains statements
                assert item[
                           f"rw{i + 1}_factuality"] == 0, f"Expected factuality rate to be 0 for repeated model {m} in paper {item['target_paper_id']}, but got {item[f'rw{i + 1}_factuality']}"
                continue
            res[item["target_paper_id"]][m] = item[f"rw{i + 1}_factuality"]

    list_res = []
    for target_paper_id, factuality_rates in res.items():
        x = {
            "target_paper_id": target_paper_id,
        }
        for model, factuality_rate in factuality_rates.items():
            x[model] = factuality_rate
        list_res.append(x)

    dataset = Dataset.from_list(list_res)
    dataset.to_json(args.output, orient="records", lines=True)


def factuality_win_rate(args):
    if args.dataset is None:
        print("Error: --dataset argument is required for factuality_win_rate command.", file=sys.stderr)
        sys.exit(1)

    wins = defaultdict(int)
    ties = defaultdict(int)
    total_duels = 0
    with open(args.dataset, "r") as f:
        for line in f:
            record = json.loads(line)

            # get max factuality rate
            models = [k for k in record if k != "target_paper_id"]

            for duel in itertools.combinations(models, 2):
                total_duels += 1
                first_model, second_model = duel[0], duel[1]

                if record[first_model] > record[second_model]:
                    wins[first_model] += 1
                elif record[second_model] > record[first_model]:
                    wins[second_model] += 1
                else:
                    ties[first_model] += 1
                    ties[second_model] += 1

    print(f"Total duels: {total_duels}")
    for model, win_count in wins.items():
        win_rate = win_count / total_duels if total_duels > 0 else 0
        print(
            f"Model: {model}, Win Count: {win_count}, Win Rate: {win_rate:.4f}, Tie Count: {ties[model]}, Tie Rate: {ties[model] / total_duels if total_duels > 0 else 0:.4f}")


def create_config(args):
    if args.workflow not in ["coarse_meta_eval", "statement_meta_eval"]:
        print(f"Error: Unknown workflow '{args.workflow}'. Available workflows: coarse_meta_eval, statement_meta_eval",
              file=sys.stderr)
        sys.exit(1)

    if args.workflow == "coarse_meta_eval":
        config = Config(MetaEvalCoarseWorkflow)
    elif args.workflow == "statement_meta_eval":
        config = Config(MetaEvalStatementWorkflow)
    else:
        raise NotImplementedError(f"Workflow '{args.workflow}' is not implemented yet.")

    config.save(sys.stdout)

    if args.doc:
        with open(args.doc, "w") as f:
            f.write(config.generate_md_documentation())


def coarse_meta_eval(args):
    workflow = MetaEvalCoarseWorkflow.create(args.config)
    workflow()


def statement_meta_eval(args):
    workflow = MetaEvalStatementWorkflow.create(args.config)
    workflow()


def human_classification_baseline(args):
    if args.dataset is not None and Path(args.dataset).exists():
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(
            "BUT-FIT/OARelatedWorkMetaEval",
            split="statements"
        )

    annotator_id = args.annotator

    for item in dataset:
        label = json.loads(item["label"])
        print(json.dumps({
            "id": item["id"],
            "classification": label[annotator_id],
        }))


def statement_stats(args):
    if args.dataset is not None and Path(args.dataset).exists():
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(
            "BUT-FIT/OARelatedWorkMetaEval",
            split="statements"
        )

    # Load the English NLP model for syntactic parsing
    nlp = spacy.load("en_core_web_sm")

    def get_parse_tree_depth(node, current_depth):
        """Recursively walks the dependency tree to find the maximum depth."""
        if node.n_lefts + node.n_rights > 0:
            return max(get_parse_tree_depth(child, current_depth + 1) for child in node.children)
        return current_depth

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_verification_distance(statement, sources):
        if not isinstance(statement, str) or not statement.strip():
            return {"Extractive Word Overlap (R1)": 0.0, "Extractive Word Overlap (R2)": 0.0, "Extractive Phrase Overlap (RL)": 0.0}

        # If sources is a list of multiple documents, join them into one giant reference text
        if isinstance(sources, list):
            reference_text = " ".join([str(s) for s in sources if s])
        else:
            reference_text = str(sources)

        # If there are no sources, return 0
        if not reference_text.strip():
            return {"Extractive Word Overlap (R1)": 0.0,  "Extractive Word Overlap (R2)": 0.0, "Extractive Phrase Overlap (RL)": 0.0}

        # Calculate ROUGE scores
        # Note: ROUGE usually puts the reference first, then the prediction.
        # Here, 'reference' is the source text, 'prediction' is the statement.
        scores = scorer.score(reference_text, statement)

        # We specifically want PRECISION:
        # (Number of overlapping words) / (Total words in the STATEMENT)
        # This tells us exactly what percentage of the statement was copy-pasted.
        r1_precision = scores['rouge1'].precision
        r2_precision = scores['rouge2'].precision
        rl_precision = scores['rougeL'].precision

        return {
            "Extractive Word Overlap (R1)": r1_precision,
            "Extractive Word Overlap (R2)": r2_precision,
            "Extractive Phrase Overlap (RL)": rl_precision
        }
    def analyze_statement(text, sources):
        if not isinstance(text, str) or not text.strip():
            return {
                "Word Count": 0,
                "Flesch-Kincaid Grade": 0.0,
                "Flesch Reading Ease": 0.0,
                "Max Parse Tree Depth": 0,
                "Extractive Word Overlap (R1)": 0.0,
                "Extractive Word Overlap (R2)": 0.0,
                "Extractive Phrase Overlap (RL)": 0.0,
                "Citation Count": 0
            }

        # 1. Word Count
        word_count = len(text.split())

        # 2. Flesch-Kincaid Grade Level
        fk_grade = textstat.flesch_kincaid_grade(text)
        f_ease = textstat.flesch_reading_ease(text)

        # 3. Maximum Syntactic Parse Tree Depth (Grammar complexity)
        doc = nlp(text)
        max_depth = 0
        for sent in doc.sents:
            depth = get_parse_tree_depth(sent.root, 0)
            if depth > max_depth:
                max_depth = depth

        # 4. Citation Count (Matches "[1]", "[1, 2]", or "(Author, 2023)")
        # Customize these regex patterns if your dataset uses a different citation format
        citations_bracket = len(re.findall(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', text))
        citations_paren = len(re.findall(r'\([A-Za-z\s]+,\s*\d{4}\)', text))
        citation_count = citations_bracket + citations_paren

        overlaps = calculate_verification_distance(text, sources)

        return {
            "Word Count": word_count,
            "Flesch-Kincaid Grade": fk_grade,
            "Flesch Reading Ease": f_ease,
            "Max Parse Tree Depth": max_depth,
            "Extractive Word Overlap (R1)": overlaps["Extractive Word Overlap (R1)"],
            "Extractive Word Overlap (R2)": overlaps["Extractive Word Overlap (R2)"],
            "Extractive Phrase Overlap (RL)": overlaps["Extractive Phrase Overlap (RL)"],
            "Citation Count": citation_count
        }

    random_subset_stats = defaultdict(int)
    adversarial_subset_stats = defaultdict(int)
    original_human_subset_stats = defaultdict(int)
    all_human_differ_subset_stats = defaultdict(int)
    unverifiable_to_verifiable_stats = defaultdict(int)
    unverifiable_cnt = defaultdict(int)
    unverifiable_consensual_cnt = defaultdict(int)
    consensual_unverifiable_to_verifiable_stats = defaultdict(int)
    changes_cnt = defaultdict(lambda : defaultdict(float))
    model_cnts = defaultdict(int)
    atomic_claims_cnt = defaultdict(lambda: defaultdict(int))
    consensual_unverifiable_atomic_claims_cnt = defaultdict(lambda: defaultdict(int))

    linquistic_stats = defaultdict(lambda : defaultdict(list))
    linquistic_stats_for_consensual_unverifiable = defaultdict(lambda : defaultdict(list))

    statement_ids_2_claims = None

    if args.atomic_claims is not None:
        with open(args.atomic_claims, "r") as f:
            statement_ids_2_claims = {int(record["id"]): record["claims"] for record in map(json.loads, f)}

    for item in dataset:

        model = item["statement_id"].split("_", maxsplit=1)[1]
        model = model.rsplit("_", maxsplit=1)[0]
        model_cnts[model] += 1

        lin_stats = analyze_statement(item["text"], (item["txt_target_paper"] + " " + item["txt_cited_papers"]).replace("\n", " ")) # not sure how \n are handled backing off to safe variant
        for k, v in lin_stats.items():
            linquistic_stats[model][k].append(v)

        label = json.loads(item["label"])
        if item["differ"]:
            adversarial_subset_stats[model] += 1
        else:
            random_subset_stats[model] += 1

        if label["82"] != label["83"]:
            original_human_subset_stats[model] += 1

        if len(set([label["82"], label["83"], label["meta"]])) > 1:
            all_human_differ_subset_stats[model] += 1

        if statement_ids_2_claims:
            claim_count = len(statement_ids_2_claims[item["id"]])
            atomic_claims_cnt[model][claim_count] += 1
            atomic_claims_cnt["all"][claim_count] += 1

        changes_cnt[model][(label["82"], label["83"], label["meta"])] += 1
        changes_cnt["all"][(label["82"], label["83"], label["meta"])] += 1

        if label["82"] == "Unverifiable" or label["83"] == "Unverifiable":
            unverifiable_cnt[model] += 1
            if label["82"] == "Unverifiable" and label["83"] == "Unverifiable":
                unverifiable_consensual_cnt[model] += 1
                for k, v in lin_stats.items():
                    linquistic_stats_for_consensual_unverifiable[model][k].append(v)

                if statement_ids_2_claims:
                    claim_count = len(statement_ids_2_claims[item["id"]])
                    consensual_unverifiable_atomic_claims_cnt[model][claim_count] += 1
                    consensual_unverifiable_atomic_claims_cnt["all"][claim_count] += 1

            if label["meta"] != "Unverifiable":
                unverifiable_to_verifiable_stats[model] += 1

                if label["82"] == "Unverifiable" and label["83"] == "Unverifiable":
                    consensual_unverifiable_to_verifiable_stats[model] += 1

    print("Linquistic stats:")
    for model, res in linquistic_stats.items():
        print(f"  Model: {model}")
        for k, v in res.items():
            print(f"    {k}: {sum(v)/len(v) if len(v) > 0 else 0:.4f}")

    print("Linquistic stats for consensual unverifiable:")
    for model, res in linquistic_stats_for_consensual_unverifiable.items():
        print(f"  Model: {model}")
        for k, v in res.items():
            print(f"    {k}: {sum(v)/len(v) if len(v) > 0 else 0:.4f}")

    print("Adversarial subset stats:")
    for model, count in adversarial_subset_stats.items():
        print(f"  Model: {model}, Count: {count} / {model_cnts[model]} ({count/model_cnts[model]})")

    print("Original human differ subset stats:")
    for model, count in original_human_subset_stats.items():
        print(f"  Model: {model}, Count: {count} / {model_cnts[model]} ({count/model_cnts[model]})")

    print("All human differ subset stats:")
    for model, count in all_human_differ_subset_stats.items():
        print(f"  Model: {model}, Count: {count} / {model_cnts[model]} ({count/model_cnts[model]})")

    print("Unverifiable to verifiable stats:")
    for model, count in unverifiable_to_verifiable_stats.items():
        print(f"  Model: {model}, Count: {count} / {model_cnts[model]} ({count/model_cnts[model]}), out of unverifiable {count} / {unverifiable_cnt[model]} ({count/unverifiable_cnt[model]})")

    print("Consensual unverifiable to verifiable stats:")
    for model, count in consensual_unverifiable_to_verifiable_stats.items():
        print(f"  Model: {model}, Count: {count} / {model_cnts[model]} ({count/model_cnts[model]}), out of unverifiable consensual {count} / {unverifiable_consensual_cnt[model]} ({count/unverifiable_consensual_cnt[model]})")

    print("Changes stats:")
    for model, changes in sorted(changes_cnt.items(), key=lambda x: x[0], reverse=True):
        print(f"  Model: {model} ({sum(changes.values())})")
        for change, count in sorted(changes.items(), key=lambda x: x[1], reverse=True):
            print(f"    {change[0]} {change[1]} => {change[2]} = {count}")

        differ = {change: changes[change] for change in changes if len(set(change)) > 1}
        print(f"  Differ Model: {model} ({sum(differ.values())})")
        for change, count in sorted(differ.items(), key=lambda x: x[1], reverse=True):
            print(f"    {change[0]} {change[1]} => {change[2]} = {count}")

        unverifiable_to_verifiable_soft = sum(count for change, count in changes.items() if "Unverifiable" in change[:2] and change[2] != "Unverifiable")
        unverifiable_to_verifiable_consensual = sum(count for change, count in changes.items() if "Unverifiable" == change[0] and "Unverifiable" == change[1] and change[2] != "Unverifiable")
        print(f"  Unverifiable to verifiable soft (including non consensual): {model} = {unverifiable_to_verifiable_soft}")
        print(f"  Unverifiable to verifiable consensual: {model} = {unverifiable_to_verifiable_consensual}")

    if statement_ids_2_claims is not None:
        print("Atomic claims mean stats:")
        for model in sorted(atomic_claims_cnt):
            freq_by_claim_count = atomic_claims_cnt[model]
            total_statements = sum(freq_by_claim_count.values())

            all_values = []
            for count, freq in freq_by_claim_count.items():
                all_values.extend([count] * freq)

            mean_claim_count = (
                sum(all_values) / total_statements
                if total_statements > 0 else 0
            )

            median_claim_count = statistics.median(all_values) if all_values else 0

            print(
                f"  Model: {model}, Mean atomic claims: {mean_claim_count:.4f}, Median atomic claims {median_claim_count:.4f}, Total statements: {total_statements}"
            )

        print("Atomic claims histogram:")
        for model in sorted(atomic_claims_cnt):
            print(f"  Model: {model}")
            bars = [
                (str(claim_count), frequency)
                for claim_count, frequency in sorted(atomic_claims_cnt[model].items())
            ]
            if bars:
                print_histogram(bars)
            else:
                print("    No atomic-claim data.")

        print("Atomic claims mean stats for consensual unverifiable statements:")
        for model in sorted(consensual_unverifiable_atomic_claims_cnt):
            freq_by_claim_count = consensual_unverifiable_atomic_claims_cnt[model]
            total_statements = sum(freq_by_claim_count.values())

            all_values = []
            for claim_count, frequency in freq_by_claim_count.items():
                all_values.extend([claim_count] * frequency)

            mean_claim_count = (
                sum(all_values) / total_statements
                if total_statements > 0 else 0
            )

            median_claim_count = statistics.median(all_values) if all_values else 0

            print(
                f"  Model: {model}, Mean atomic claims: {mean_claim_count:.4f}, Median atomic claims {median_claim_count:.4f}, Total statements: {total_statements}"
            )

        print("Atomic claims histogram for consensual unverifiable statements:")
        for model in sorted(consensual_unverifiable_atomic_claims_cnt):
            print(f"  Model: {model}")
            bars = [
                (str(claim_count), frequency)
                for claim_count, frequency in sorted(consensual_unverifiable_atomic_claims_cnt[model].items())
            ]
            if bars:
                print_histogram(bars)
            else:
                print("    No atomic-claim data.")


def model_classification_types_stats(args):
    classification_types = defaultdict(lambda: defaultdict(int))

    if args.given_statements is None:
        with open(args.results, "r") as f:
            for line in f:
                record = json.loads(line)
                model = record["model"]

                for statement in record["statements"]:
                    if "classification" in statement:
                        classification_types[model][statement["classification"]] += 1
                    else:
                        classification_types[model][None] += 1

    else:
        if Path(args.given_statements).exists():
            dataset = load_from_disk(args.given_statements)
        else:
            dataset = load_dataset(
                "BUT-FIT/OARelatedWorkMetaEval",
                "statements",
                split="train"
            )

        statement_id_2_samples = {}

        meta_counts = defaultdict(lambda: defaultdict(int))
        for item in dataset:
            statement_id_2_samples[item["id"]] = item
            
            meta_counts[item["model"]][json.loads(item["label"])["meta"]] += 1

        with open(args.results, "r") as f:
            for line in f:
                record = json.loads(line)
                model = statement_id_2_samples[int(record["id"])]["model"]
                classification_types[model][record["classification"]] += 1

    print("Meta annotation distribution:")
    for model in sorted(meta_counts):
        print(f"  Model: {model}")
        total_statements = sum(meta_counts[model].values())
        print(f"    Total statements: {total_statements}")
        for classification, count in sorted(meta_counts[model].items(), key=lambda x: x[1], reverse=True):
            print(f"    {classification}: {count} ({count/total_statements:.4f})")

    print("Model classification types stats:")
    for model in sorted(classification_types):
        print(f"  Model: {model}")
        total_statements = sum(classification_types[model].values())
        print(f"    Total statements: {total_statements}")
        for classification, count in sorted(classification_types[model].items(), key=lambda x: x[1], reverse=True):
            print(f"    {classification}: {count} ({count/total_statements:.4f})")

    




def main():
    parser = argparse.ArgumentParser(description="OARelatedWorkMetaEval CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Dataset Creation Parser
    parser_dataset = subparsers.add_parser("dataset_creation", help="Create the dataset")
    parser_dataset.add_argument("label_studio_annotations", type=str, help="Path to Label Studio annotations")
    parser_dataset.add_argument("label_studio_input_file", type=str,
                                help="Path to Label Studio input file with model order and original generated related work")
    parser_dataset.add_argument("randomly_selected_statements", type=str,
                                help="Path to JSON file with randomly selected statements for each data and model")
    parser_dataset.add_argument("randomly_selected_differ_statements", type=str,
                                help="Path to JSON file with randomly selected statements that differ between models for each data and model")
    parser_dataset.add_argument("meta", type=str, help="Path to csv file with meta annotations")
    parser_dataset.add_argument("text_representations", type=str,
                                help="Path to file with text representations of inputs")
    parser_dataset.add_argument("go_text_representations", type=str,
                                help="Path to file with go text representations of inputs")
    parser_dataset.add_argument("output_dir", type=str, help="Output directory for the dataset")
    parser_dataset.set_defaults(func=dataset_creation)

    parser_annot_times = subparsers.add_parser("annot_times", help="Calculate annotation times")
    parser_annot_times.add_argument("label_studio_annotations", type=str, help="Path to Label Studio annotations")
    parser_annot_times.add_argument("label_studio_input_file", type=str,
                                help="Path to Label Studio input file with model order and original generated related work")
    parser_annot_times.set_defaults(func=annot_times)

    parser_model_classification_types_stats = subparsers.add_parser("model_classification_types_stats",
                                                    help="Performs meta evaluation on the coarse level")
    parser_model_classification_types_stats.add_argument("results", type=str, help="Path to file with extracted statements")
    parser_model_classification_types_stats.add_argument("-g", "--given_statements", type=str, help="If provided, path to statement dataset configuration")
    parser_model_classification_types_stats.set_defaults(func=model_classification_types_stats)

    parser_create_config = subparsers.add_parser("create_config",
                                                 help="Create yaml configuration for given workflow. Available workflows: coarse_meta_eval, statement_meta_eval")
    parser_create_config.add_argument("workflow", type=str,
                                      help="Workflow for which to create configuration. Available workflows: coarse_meta_eval, statement_meta_eval")
    parser_create_config.add_argument("--doc", type=str,
                                      help="Voluntary path where the md documentation file describing config will be saved")
    parser_create_config.set_defaults(func=create_config)

    parser_coarse_meta_eval = subparsers.add_parser("coarse_meta_eval",
                                                    help="Performs meta evaluation on the coarse level")
    parser_coarse_meta_eval.add_argument("config", type=str, help="Path to configuration file")
    parser_coarse_meta_eval.set_defaults(func=coarse_meta_eval)

    parser_statement_meta_eval = subparsers.add_parser("statement_meta_eval",
                                                    help="Performs meta evaluation on the statement level")
    parser_statement_meta_eval.add_argument("config", type=str, help="Path to configuration file")
    parser_statement_meta_eval.set_defaults(func=statement_meta_eval)

    parser_statement_stats = subparsers.add_parser("statement_stats", help="Calculate statistics about statements in the dataset")
    parser_statement_stats.add_argument("--dataset", type=str, help="Path to dataset folder")
    parser_statement_stats.add_argument("--atomic_claims", type=str, help="Path to atomic claims file. Statistics for atomic claims will be provided.")
    parser_statement_stats.set_defaults(func=statement_stats)

    # Calculate duel win-score
    duel_win_score_parser = subparsers.add_parser("duel_win_score", help="Calculate duel win-score")
    duel_win_score_parser.add_argument("--dataset", type=str, help="Path to duel results file")
    duel_win_score_parser.add_argument("--bootstrap_iterations", type=int, default=10000,
                                       help="Number of bootstrap resamples for the per-pair CI")
    duel_win_score_parser.add_argument("--bootstrap_confidence", type=float, default=0.95,
                                       help="Confidence level for the bootstrap CI (e.g. 0.95)")
    duel_win_score_parser.add_argument("--bootstrap_seed", type=int, default=0,
                                       help="Seed for the bootstrap RNG (for reproducibility)")
    duel_win_score_parser.set_defaults(func=duel_win_score)

    # Calculate statement score
    statement_score_parser = subparsers.add_parser("statement_score", help="Calculate statement score")
    statement_score_parser.add_argument("--dataset", type=str, help="Path to duel results file")
    statement_score_parser.add_argument("--output_verifiable_statements", type=str,
                                        help="Path to output file for verifiable statements")
    statement_score_parser.add_argument("--meta", action="store_true",
                                        help="Whether to use meta annotations for calculating statement score")
    statement_score_parser.set_defaults(func=statement_score)

    # Calculate statement score from model
    statement_score_from_model_initial_parser = subparsers.add_parser("statement_score_from_model_initial",
                                                              help="Calculate statement score assesst by model in format initially used for selecting statements")
    statement_score_from_model_initial_parser.add_argument("random_statements", type=str, help="Path to model outputs file")
    statement_score_from_model_initial_parser.add_argument("all_statements", type=str,
                                                   help="All statements that should were labeled")
    statement_score_from_model_initial_parser.add_argument("--verifiable_statements",
                                                   type=str,
                                                   help="If provided, calculates scores only for verifiable statements in this file")
    statement_score_from_model_initial_parser.set_defaults(func=statement_score_from_model_initial)

    statement_score_from_model_parser = subparsers.add_parser("statement_score_from_model",
                                                                      help="Calculate statement score assesst by model.")
    statement_score_from_model_parser.add_argument("random_statements", type=str,
                                                           help="Path to model outputs file")
    statement_score_from_model_parser.add_argument("all_statements", type=str,
                                                           help="All statements that should were labeled")
    statement_score_from_model_parser.add_argument("--verifiable_statements",
                                                           type=str,
                                                           help="If provided, calculates scores only for verifiable statements in this file")
    statement_score_from_model_parser.set_defaults(func=statement_score_from_model_parser)

    extract_factuality_rates_parser = subparsers.add_parser("extract_factuality_rates",
                                                            help="Extract factuality rates for each target paper and model")
    extract_factuality_rates_parser.add_argument("dataset", type=str, help="Path to dataset file")
    extract_factuality_rates_parser.add_argument("output", type=str,
                                                 help="Path to output file for factuality rates. If not provided, prints to stdout.")
    extract_factuality_rates_parser.set_defaults(func=extract_factuality_rates)

    # Calculate statement score
    factuality_win_rate_parser = subparsers.add_parser("factuality_win_rate", help="Calculate duel win score")
    factuality_win_rate_parser.add_argument("dataset", type=str, help="Path to data with factuality rates")
    factuality_win_rate_parser.set_defaults(func=factuality_win_rate)

    # create human classification baseline
    human_classification_baseline_parser = subparsers.add_parser("human_classification_baseline", help="Create human classification baseline")
    human_classification_baseline_parser.add_argument("annotator", type=str, help="Annotator id")
    human_classification_baseline_parser.add_argument("--dataset", type=str, help="Path to dataset file")
    human_classification_baseline_parser.set_defaults(func=human_classification_baseline)

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
