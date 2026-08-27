 *  Configuration for `MetaEvalStatementWorkflow`
     *  Example configuration: 
        ```yaml
        dataset_path:  # Dataset path or None to load it from HF hub
        results: [] # Files with statement classification results in jsonl format with `id` and `classification` fields.
        bootstrap_resamples: 10000 # Bootstrap number of bootstrap samples
        confidence_level: 0.95 # Confidence level for bootstrap confidence interval
        random_seed: # Random seed
        ```
     *  Attributes:
         * dataset_path
            * <b>Description:</b> Dataset path or None to load it from HF hub
            * <b>Type:</b> `Optional`
         * results
            * <b>Description:</b> Files with statement classification results in jsonl format with `id` and `classification` fields.
            * <b>Type:</b> List of subclasses of `StatementClassificationResultFile`
            * <b>Available subclasses:</b>
         * bootstrap_resamples
            * <b>Description:</b> Bootstrap number of bootstrap samples
            * <b>Type:</b> `int`
            * <b>Default value:</b> `10000`
         * confidence_level
            * <b>Description:</b> Confidence level for bootstrap confidence interval
            * <b>Type:</b> `float`
            * <b>Default value:</b> `0.95`
         * random_seed
            * <b>Description:</b> Random seed
            * <b>Type:</b> `int | None`
