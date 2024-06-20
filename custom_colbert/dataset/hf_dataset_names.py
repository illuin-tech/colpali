from enum import Enum


class TrainDatasets(Enum):
    """
    Dataset names for the training datasets used in HuggingFace Datasets.
    """

    government_reports = "coldoc/syntheticDocQA_government_reports_train"
    healthcare_industry = "coldoc/syntheticDocQA_healthcare_industry_train"
    energy = "coldoc/syntheticDocQA_energy_train"
    artificial_intelligence = "coldoc/syntheticDocQA_artificial_intelligence_train"
    arxivqa = "coldoc/arxivqa_train"
    docvqa = "coldoc/docvqa_train"
    infovqa = "coldoc/infovqa_train"
    tatqa = "coldoc/tatqa_train"

    @staticmethod
    def get_synthetic_datasets():
        return [
            TrainDatasets.government_reports,
            TrainDatasets.healthcare_industry,
            TrainDatasets.energy,
            TrainDatasets.artificial_intelligence,
        ]


class TestImagesDirpath(Enum):
    """
    Dataset names for the test datasets used in HuggingFace Datasets.
    """

    government_reports = "data/government_reports"
    healthcare_industry = "data/healthcare_industry"
    energy = "data/energy"
    artificial_intelligence = "data/scrapped_pdfs_split/pages_extracted/artificial_intelligence_test"
    arxivqa = "data/arxivqa"
    docvqa = "data/docvqa"
    infovqa = "data/infovqa"
    tatqa = "data/tatqa"


class CaptionedSyntheticDatasets(Enum):
    """
    Dataset names for the captioned synthetic datasets used in HuggingFace Datasets.
    """

    shift = "coldoc/baseline_cap_shiftproject_test"


class SyntheticDocQATest(Enum):
    shift = "coldoc/shiftproject_test"
