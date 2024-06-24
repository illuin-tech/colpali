from enum import Enum


class TrainDatasets(Enum):
    """
    Dataset names for the training datasets used in HuggingFace Datasets.
    """

    government_reports = "vidore/syntheticDocQA_government_reports_train"
    healthcare_industry = "vidore/syntheticDocQA_healthcare_industry_train"
    energy = "vidore/syntheticDocQA_energy_train"
    artificial_intelligence = "vidore/syntheticDocQA_artificial_intelligence_train"
    arxivqa = "vidore/arxivqa_train"
    docvqa = "vidore/docvqa_train"
    infovqa = "vidore/infovqa_train"
    tatqa = "vidore/tatqa_train"

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

    shift = "vidore/baseline_cap_shiftproject_test"


class SyntheticDocQATest(Enum):
    shift = "vidore/shiftproject_test"
