"""
Created on 16.02.26

:author:     Martin Dočekal
"""
import json
import re
from collections import defaultdict

from bs4 import BeautifulSoup

statements_pattern = r'rw_\d+_statement_\d+'
statements_regex = re.compile(statements_pattern)

STATEMENT_TRANSLATION = {
    283698: {"xGE0TKf0b-": "rw_1_statement_2"}    # there was one annotator that reported to me that he/she accidentally changed this id
}


def find_statements(statements, statement) -> dict | None:
    for s in statements:
        if s["statement"] == statement:
            return s
    return None


def search_result_by_id(results, id):
    for result in results:
        if result.get('id') == id:
            return result
    return None


def rendered_to_original_offset_mapping(text_for_label_studio: str, original_text: str) -> list[int]:
    """
    Create a mapping from rendered text offsets to original text offsets.

    :param text_for_label_studio: The text as code for rendeing in Label Studio, which may contain HTML tags.
    :param original_text: The original text without HTML tags.
    :return: A list where the index represents the offset in the rendered text and the value at that index is the corresponding offset in the original text.
    """
    soup = BeautifulSoup(text_for_label_studio, 'html.parser')
    rendered_text = soup.get_text()

    mapping = []
    original_idx = 0
    for i, char in enumerate(rendered_text):
        while original_text[original_idx] != char and original_idx < len(original_text):
            original_idx += 1
        if original_idx < len(original_text):
            mapping.append(original_idx)
            original_idx += 1
        else:
            raise ValueError(f"Rendered text contains character '{char}' at position {i} that is not found in the original text starting from position {original_idx}.")

    return mapping


def load_and_prepare_annotations(export_path: str, model_order: list, randomly_selected_statements: dict, randomly_selected_differ_statements: dict, label_studio_input: list) -> dict:
    """
    Load and prepare data from Label Studio export.

    :param export_path: Path to the Label Studio export JSON file.
    :param model_order: List of model IDs in the order they were presented to annotators for each task.
    :param randomly_selected_statements: Dictionary mapping from data ID and model ID to the list of randomly selected statements for that data and model. The key is in the form "{data_id}_{model_id}".
    :param randomly_selected_differ_statements: Dictionary mapping from data ID and model ID to the list of randomly selected statements that differ between models for that data and model. The key is in the form "{data_id}_{model_id}".
    :param label_studio_input: The original input data for Label Studio, which contains the raw text for rw_1 and rw_2 for each task. This is needed to create the mapping from rendered offsets to original offsets.
    :return: Dictionary containing annotation results structured as {task_id:
    """

    with open(export_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Structure to hold annotation results: {target_paper_id: {task_id: {annotator_id: label}}}
    annotator_results = defaultdict(lambda: defaultdict(dict))

    # Structure to hold the set of all annotators encountered
    all_annotator_ids = set()

    for i, task in enumerate(data):
        task_id = task['id']
        task_model_order = model_order[i]
        orig_target_paper_id = task["data"]["id"]

        rw_1_offset_mapping = rendered_to_original_offset_mapping(task["data"]["rw_1"], label_studio_input[i]["data"]["orig_rw_1"])
        rw_2_offset_mapping = rendered_to_original_offset_mapping(task["data"]["rw_2"], label_studio_input[i]["data"]["orig_rw_2"])

        annotator_results[orig_target_paper_id][task_id]["annot_rw_1"] = task["data"]["rw_1"]
        annotator_results[orig_target_paper_id][task_id]["annot_rw_2"] = task["data"]["rw_2"]
        annotator_results[orig_target_paper_id][task_id]["annot_target_paper"] = task["data"]["target_paper"]
        annotator_results[orig_target_paper_id][task_id]["annot_cited_papers"] = task["data"]["cited_papers"]
        annotator_results[orig_target_paper_id][task_id]["model_order"] = task_model_order
        annotator_results[orig_target_paper_id][task_id]["annotations"] = {
            "model": {
                        "statements": {},
                        "randomly_selected_statements": {},
                        "randomly_selected_differ_statements": {}
            }
        }

        # Iterate over all annotations for the current task
        for annotation in task.get('annotations', []):
            annotator_id = annotation.get('completed_by')  # Use user ID who completed it
            results = annotation.get('result', [])

            # Find the result for the specific choices tag
            for result in results:
                if annotator_id not in annotator_results[orig_target_paper_id][task_id]["annotations"]:
                    annotator_results[orig_target_paper_id][task_id]["annotations"][annotator_id] = {
                        "preference": None,
                        "relevance": None,
                        "faithfulness": None,
                        "language": None,
                        "statements": {},
                        "randomly_selected_statements": {},
                        "randomly_selected_differ_statements": {}
                    }

                if result.get("type") == "relation":  # check that there is no label missmatch
                    """
                    {
                    "type": "relation",
                    "to_id": "rw_1_statement_2",
                    "from_id": "yf_pkDr3As",
                    "direction": "right"
                    }

                    """

                    if statements_regex.match(result.get('to_id', '')) or statements_regex.match(
                            result.get('from_id', '')):
                        to_label = search_result_by_id(results, result.get('to_id', ''))['value']['labels'][0]
                        from_label = search_result_by_id(results, result.get('from_id', ''))['value']['labels'][0]
                        if to_label != from_label:
                            print(
                                f"Label missmatch for task {task_id} and annotator {annotator_id} for relation {result.get('to_id', '')} and {result.get('from_id', '')}")

                if result.get('from_name') == "place_rw_2" and result.get('type') == 'choices':
                    # The value is typically a list, take the first choice
                    value = result['value']['choices'][0]

                    # Store the label
                    annotator_results[orig_target_paper_id][task_id]["annotations"][annotator_id]["preference"] = value

                    all_annotator_ids.add(annotator_id)

                if result.get("from_name") == "relevance_rating" and result.get('type') == 'choices':
                    # The value is typically a list, take the first choice
                    value = result['value']['choices'][0]

                    # Store the label
                    annotator_results[orig_target_paper_id][task_id]["annotations"][annotator_id]["relevance"] = value

                    all_annotator_ids.add(annotator_id)

                if result.get("from_name") == "faithfulness_rating" and result.get('type') == 'choices':
                    # The value is typically a list, take the first choice
                    value = result['value']['choices'][0]

                    # Store the label
                    annotator_results[orig_target_paper_id][task_id]["annotations"][annotator_id]["faithfulness"] = value

                    all_annotator_ids.add(annotator_id)

                if result.get("from_name") == "language_rating" and result.get('type') == 'choices':
                    # The value is typically a list, take the first choice
                    value = result['value']['choices'][0]

                    # Store the label
                    annotator_results[orig_target_paper_id][task_id]["annotations"][annotator_id]["language"] = value

                    all_annotator_ids.add(annotator_id)

                # take statements
                # id in form: rw_{number}_statement_{number}

                statement_id = result.get('id', '')
                if task_id in STATEMENT_TRANSLATION:
                    if statement_id in STATEMENT_TRANSLATION[task_id]:
                        statement_id = STATEMENT_TRANSLATION[task_id][statement_id]

                if statements_regex.match(statement_id):
                    offset_mapping = rw_1_offset_mapping if statement_id.startswith("rw_1") else rw_2_offset_mapping
                    rw_number = int(statement_id.split('_')[1])
                    model_id = task_model_order[rw_number - 1]

                    # Get original statement positions
                    orig_start = result['value'].get('start')
                    orig_end = result['value'].get('end')

                    # reported problems fixes
                    if task_id in STATEMENT_TRANSLATION and result.get('id', '') in STATEMENT_TRANSLATION[task_id]:
                        orig_start = result['value']["globalOffsets"]["start"]
                        orig_end = result['value']["globalOffsets"]["end"]

                    if result['value']['labels'][0] == "Unlabeled" and annotator_id == 83 and task_id == 339811:
                        # this is known issue, that there is one annotation with having unlabeled, but the referenced assiciated edited span is false
                        result['value']['labels'][0] = "False"

                    # Find relation for this statement to get edited span positions
                    edited_start = None
                    edited_end = None

                    # Look for relations where this statement is the TO item
                    for rel_result in results:
                        if rel_result.get('type') == 'relation' and (rel_result.get('to_id') == statement_id or rel_result.get('from_id') == statement_id):
                            # Found a relation, get the FROM item (edited span)
                            associated_id = rel_result.get('from_id') if statements_regex.match(rel_result.get('to_id')) else rel_result.get('to_id')
                            edited_span = search_result_by_id(results, associated_id)
                            if statements_regex.match(associated_id):
                                # we are not interested in this relation
                                continue
                            if edited_span and 'value' in edited_span:
                                edited_start = edited_span['value']["globalOffsets"]["start"]
                                edited_end = edited_span['value']["globalOffsets"]["end"]
                                break

                    search_rand_sel_statement = find_statements(randomly_selected_statements[str(orig_target_paper_id) + "_" + str(model_id)], result['value']['text'])
                    search_ran_sel_dif_statement = find_statements(randomly_selected_differ_statements[str(orig_target_paper_id) + "_" + str(model_id)], result['value']['text'])

                    annotator_results[orig_target_paper_id][task_id]["annotations"][annotator_id]["statements"][statement_id] = {
                        "label": result['value']['labels'][0],
                        "text": result['value']['text'],
                        "start": offset_mapping[orig_start] if orig_start is not None else None,
                        "end": offset_mapping[orig_end - 1] + 1 if orig_end is not None else None,
                        "edited_start": offset_mapping[edited_start] if edited_start is not None else None,
                        "edited_end": offset_mapping[edited_end - 1] + 1 if edited_end is not None else None,
                        "rendered_start": orig_start,
                        "rendered_end": orig_end,
                        "rendered_edited_start": edited_start,
                        "rendered_edited_end": edited_end
                    }
                    annotator_results[orig_target_paper_id][task_id]["annotations"]["model"]["statements"][
                        statement_id] = {
                        "label": (search_ran_sel_dif_statement if search_rand_sel_statement is None else search_rand_sel_statement)["is_true"],
                        "text": result['value']['text'],
                        "start": offset_mapping[orig_start] if orig_start is not None else None,
                        "end": offset_mapping[orig_end - 1] + 1 if orig_end is not None else None,
                        "edited_start": offset_mapping[edited_start] if edited_start is not None else None,
                        "edited_end": offset_mapping[edited_end - 1] + 1 if edited_end is not None else None,
                        "rendered_start": orig_start,
                        "rendered_end": orig_end,
                        "rendered_edited_start": edited_start,
                        "rendered_edited_end": edited_end,
                        "evidence": (search_ran_sel_dif_statement if search_rand_sel_statement is None else search_rand_sel_statement)["evidence"]
                    }

                    # is this in randomly_selected_statements?
                    if search_rand_sel_statement is not None:
                        annotator_results[orig_target_paper_id][task_id]["annotations"][annotator_id]["randomly_selected_statements"][statement_id] = {
                            "label": result['value']['labels'][0],
                            "text": result['value']['text'],
                            "start": offset_mapping[orig_start] if orig_start is not None else None,
                            "end": offset_mapping[orig_end - 1] + 1 if orig_end is not None else None,
                            "edited_start": offset_mapping[edited_start] if edited_start is not None else None,
                            "edited_end": offset_mapping[edited_end - 1] + 1 if edited_end is not None else None,
                            "rendered_start": orig_start,
                            "rendered_end": orig_end,
                            "rendered_edited_start": edited_start,
                            "rendered_edited_end": edited_end
                        }
                        annotator_results[orig_target_paper_id][task_id]["annotations"]["model"][
                            "randomly_selected_statements"][statement_id] = {
                            "label": search_rand_sel_statement["is_true"],
                            "text": result['value']['text'],
                            "start": offset_mapping[orig_start] if orig_start is not None else None,
                            "end": offset_mapping[orig_end - 1] + 1 if orig_end is not None else None,
                            "edited_start": offset_mapping[edited_start] if edited_start is not None else None,
                            "edited_end": offset_mapping[edited_end - 1] + 1 if edited_end is not None else None,
                            "rendered_start": orig_start,
                            "rendered_end": orig_end,
                            "rendered_edited_start": edited_start,
                            "rendered_edited_end": edited_end,
                            "evidence": search_rand_sel_statement["evidence"]
                        }

                    elif search_ran_sel_dif_statement is not None:
                        annotator_results[orig_target_paper_id][task_id]["annotations"][annotator_id]["randomly_selected_differ_statements"][statement_id] = {
                            "annotator": annotator_id,
                            "label": result['value']['labels'][0],
                            "text": result['value']['text'],
                            "start": offset_mapping[orig_start] if orig_start is not None else None,
                            "end": offset_mapping[orig_end - 1] + 1 if orig_end is not None else None,
                            "edited_start": offset_mapping[edited_start] if edited_start is not None else None,
                            "edited_end": offset_mapping[edited_end - 1] + 1 if edited_end is not None else None,
                            "rendered_start": orig_start,
                            "rendered_end": orig_end,
                            "rendered_edited_start": edited_start,
                            "rendered_edited_end": edited_end
                        }
                        annotator_results[orig_target_paper_id][task_id]["annotations"]["model"][
                            "randomly_selected_differ_statements"][statement_id] = {
                            "annotator": annotator_id,
                            "label": search_ran_sel_dif_statement["is_true"],
                            "text": result['value']['text'],
                            "start": offset_mapping[orig_start] if orig_start is not None else None,
                            "end": offset_mapping[orig_end - 1] + 1 if orig_end is not None else None,
                            "edited_start": offset_mapping[edited_start] if edited_start is not None else None,
                            "edited_end": offset_mapping[edited_end - 1] + 1 if edited_end is not None else None,
                            "rendered_start": orig_start,
                            "rendered_end": orig_end,
                            "rendered_edited_start": edited_start,
                            "rendered_edited_end": edited_end,
                            "evidence": search_ran_sel_dif_statement["evidence"]
                        }
                    else:
                        raise ValueError(
                            f"Statement {result['value']['text']} not found in randomly_selected_statements or randomly_selected_differ_statements for task {task_id} and model {model_id}")

    # correct missing statements

    for orig_target_paper_id, results in annotator_results.items():
        for task_id, annotations in results.items():
            annotations = annotations["annotations"]
            all_statement_ids = set(
                statement_id for annotator in annotations.values() for statement_id in annotator["statements"].keys())
            for annotator_id, annotation in annotations.items():
                for statement_id in all_statement_ids:
                    if statement_id not in annotation["statements"]:
                        raise ValueError(f"Missing statement {statement_id} for annotator {annotator_id} in task {task_id} and target paper {orig_target_paper_id}")

    return annotator_results
