 *  Configuration for `MetaEvalCoarseWorkflow`
     *  Example configuration: 
        ```yaml
        dataset_path:  # Dataset path or None to load it from HF hub
        metrics: [] # List of metrics to evaluate
        results_file: coarse_results.jsonl # Path to save the scores and correlation for each sample
        violin_plot_file: coarse_violin_plot.pdf # Path to save the violin plot of correlations
        bootstrap_resamples: 10000 # Bootstrap number of bootstrap samples
        confidence_level: 0.95 # Confidence level for bootstrap confidence interval
        random_seed: # Random seed
        reference_based_color: '#ff7f0e' # Reference based color for violin plot
        reference_free_color: '#1f77b4' # Reference free color for violin plot
        ```
     *  Attributes:
         * dataset_path
            * <b>Description:</b> Dataset path or None to load it from HF hub
            * <b>Type:</b> `Optional`
         * metrics
            * <b>Description:</b> List of metrics to evaluate
            * <b>Type:</b> List of subclasses of `Metric`
            * <b>Available subclasses:</b>
                 *  Configuration for `BertScore`
                     *  Example configuration: 
                        ```yaml
                        name:  # Name of metric
                        sample_results_file: # Path to file with detailed results for each sample
                        reference_free: # Whether the metric is reference free or reference based
                        ```
                     *  Attributes:
                         * name
                            * <b>Description:</b> Name of metric
                            * <b>Type:</b> `str`
                         * sample_results_file
                            * <b>Description:</b> Path to file with detailed results for each sample
                            * <b>Type:</b> `str`
                         * reference_free
                            * <b>Description:</b> Whether the metric is reference free or reference based
                            * <b>Type:</b> `bool`
                 *  Configuration for `BlockMatching`
                     *  Example configuration: 
                        ```yaml
                        name:  # Name of metric
                        sample_results_file: # Path to file with detailed results for each sample
                        reference_free: # Whether the metric is reference free or reference based
                        metric: ROUGE_2_F1 # Base metric for block matching. It is expected to be a metric operating on plain string level.
                        ```
                     *  Attributes:
                         * name
                            * <b>Description:</b> Name of metric
                            * <b>Type:</b> `str`
                         * sample_results_file
                            * <b>Description:</b> Path to file with detailed results for each sample
                            * <b>Type:</b> `str`
                         * reference_free
                            * <b>Description:</b> Whether the metric is reference free or reference based
                            * <b>Type:</b> `bool`
                         * metric
                            * <b>Description:</b> Base metric for block matching. It is expected to be a metric operating on plain string level.
                            * <b>Type:</b> `BaseMetric`
                            * <b>Default value:</b> `ROUGE_2_F1`
                 *  Configuration for `FromFile`
                     *  Example configuration: 
                        ```yaml
                        name:  # Name of metric
                        sample_results_file: # Path to file with detailed results for each sample
                        reference_free: # Whether the metric is reference free or reference based
                        score_file: # Loads scores for each summary from this file. It expects jsonl with fields: target_paper_id, model, score
                        ```
                     *  Attributes:
                         * name
                            * <b>Description:</b> Name of metric
                            * <b>Type:</b> `str`
                         * sample_results_file
                            * <b>Description:</b> Path to file with detailed results for each sample
                            * <b>Type:</b> `str`
                         * reference_free
                            * <b>Description:</b> Whether the metric is reference free or reference based
                            * <b>Type:</b> `bool`
                         * score_file
                            * <b>Description:</b> Loads scores for each summary from this file. It expects jsonl with fields: target_paper_id, model, score
                            * <b>Type:</b> `str`
                 *  Configuration for `Rouge`
                     *  Example configuration: 
                        ```yaml
                        name:  # Name of metric
                        sample_results_file: # Path to file with detailed results for each sample
                        reference_free: # Whether the metric is reference free or reference based
                        rouge_type: ROUGE_1 # The type of ROUGE metric: ROUGE_1, ROUGE_2, ROUGE_L, ROUGE_LSUM
                        score_component: FMEASURE # Which ROUGE score component to use: PRECISION, RECALL, FMEASURE
                        ```
                     *  Attributes:
                         * name
                            * <b>Description:</b> Name of metric
                            * <b>Type:</b> `str`
                         * sample_results_file
                            * <b>Description:</b> Path to file with detailed results for each sample
                            * <b>Type:</b> `str`
                         * reference_free
                            * <b>Description:</b> Whether the metric is reference free or reference based
                            * <b>Type:</b> `bool`
                         * rouge_type
                            * <b>Description:</b> The type of ROUGE metric: ROUGE_1, ROUGE_2, ROUGE_L, ROUGE_LSUM
                            * <b>Type:</b> `RougeType`
                            * <b>Default value:</b> `ROUGE_1`
                         * score_component
                            * <b>Description:</b> Which ROUGE score component to use: PRECISION, RECALL, FMEASURE
                            * <b>Type:</b> `ScoreComponent`
                            * <b>Default value:</b> `FMEASURE`
         * results_file
            * <b>Description:</b> Path to save the scores and correlation for each sample
            * <b>Type:</b> `str`
            * <b>Default value:</b> `coarse_results.jsonl`
         * violin_plot_file
            * <b>Description:</b> Path to save the violin plot of correlations
            * <b>Type:</b> `str`
            * <b>Default value:</b> `coarse_violin_plot.pdf`
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
         * reference_based_color
            * <b>Description:</b> Reference based color for violin plot
            * <b>Type:</b> `str`
            * <b>Default value:</b> `#ff7f0e`
         * reference_free_color
            * <b>Description:</b> Reference free color for violin plot
            * <b>Type:</b> `str`
            * <b>Default value:</b> `#1f77b4`
