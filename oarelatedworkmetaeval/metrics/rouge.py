from enum import Enum

from classconfig import ConfigurableValue
from classconfig.transforms import EnumTransformer

from oarelatedworkmetaeval.metrics.base import Metric
from oarelatedworkmetaeval.metrics.huggingfacerouge.rouge import Rouge as HFRouge


class RougeType(Enum):
    ROUGE_1 = "rouge1"
    ROUGE_2 = "rouge2"
    ROUGE_L = "rougeL"
    ROUGE_LSUM = "rougeLsum"


class ScoreComponent(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    FMEASURE = "fmeasure"


class Rouge(Metric):
    rouge_type: RougeType = ConfigurableValue(
        desc="The type of ROUGE metric: ROUGE_1, ROUGE_2, ROUGE_L, ROUGE_LSUM",
        user_default=RougeType.ROUGE_1.name,
        transform=EnumTransformer(
            RougeType
        )
    )

    score_component: ScoreComponent = ConfigurableValue(
        desc="Which ROUGE score component to use: PRECISION, RECALL, FMEASURE",
        user_default=ScoreComponent.FMEASURE.name,
        transform=EnumTransformer(ScoreComponent),
    )

    def __post_init__(self):
        self.rouge = HFRouge(allow_aggregate=False)
        self.rouge.verbose = False

    def __call__(
            self,
            target_paper_id: str,
            gen_model: str,
            summary: str,
            reference: str | None = None,
            target_paper: str | None = None,
            cited_papers: str | None = None) -> float:

        result = self.rouge.compute(
            predictions=[summary],
            references=[target_paper + "\n\n" + cited_papers] if self.reference_free else [reference],
            rouge_types=[self.rouge_type.value]
        )

        value = result[f"{self.rouge_type.value}_{self.score_component.value}"]

        if self.sample_results_file is not None:
            with open(self.sample_results_file, 'a') as f:
                print(
                    {
                        'target_paper_id': target_paper_id,
                        'model': gen_model,
                        'score': value,
                        'summary': summary,
                        'reference': target_paper + "\n\n" + cited_papers if self.reference_free else reference
                    }, file=f
                )

        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError(f"Empty ROUGE output for key '{self.rouge_type.value}_{self.score_component.value}'.")
            return float(value[0])

        return float(value)

