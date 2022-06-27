from genericpath import exists
import os
from pathlib import Path
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = ""
_DESCRIPTION = """\
"""


class COVID19Config(datasets.BuilderConfig):
    """BuilderConfig for Covid19"""

    def __init__(self, **kwargs):
        """BuilderConfig for Covid19.

        Args:
        **kwargs: keyword arguments forwarded to super.
        """
        super(COVID19Config, self).__init__(**kwargs)


class COVID19(datasets.GeneratorBasedBuilder):
    """COVID19 NER dataset."""

    BUILDER_CONFIGS = [
        COVID19Config(name="PhoNER_COVID19", version=datasets.Version("1.0.0"),
                      description="PhoNER_COVID19 dataset"),
    ]

    #PATIENT_ID, NAME, GENDER, AGE, JOB, LOCATION,ORGANIZATION, DATE, SYMPTOM_AND_DISEASE, TRANSPORTATION
    def __init__(self,
                 *args,
                 cache_dir,
                 url="https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/",
                 train_file="train_word.conll",
                 val_file="dev_word.conll",
                 test_file="test_word.conll",
                 ner_tags=('B-AGE',
                           'B-DATE',
                           'B-GENDER',
                           'B-JOB',
                           'B-LOCATION',
                           'B-NAME',
                           'B-ORGANIZATION',
                           'B-PATIENT_ID',
                           'B-SYMPTOM_AND_DISEASE',
                           'B-TRANSPORTATION',
                           'I-AGE',
                           'I-DATE',
                           'I-GENDER',
                           'I-JOB',
                           'I-LOCATION',
                           'I-NAME',
                           'I-ORGANIZATION',
                           'I-PATIENT_ID',
                           'I-SYMPTOM_AND_DISEASE',
                           'I-TRANSPORTATION',
                           'O'),
                 **kwargs):
        self._ner_tags = ner_tags
        self._url = url
        self._train_file = train_file
        self._val_file = val_file
        self._test_file = test_file
        super(COVID19, self).__init__(*args, cache_dir=cache_dir, **kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=sorted(list(self._ner_tags))
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,

        )

    def _split_generators(self, dl_manager):
        """Return SplitGenerators."""
        urls_to_download = {
            "train": f"{self._url}{self._train_file}",
            "dev": f"{self._url}{self._val_file}",
            "test": f"{self._url}{self._test_file}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                    "filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                                    "filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # SROIE2019 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            if len(tokens) > 0:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }


class COVID19Dataset(object):
    """
    """
    NAME = "PhoNER_COVID19"

    def __init__(self):
        cache_dir = os.path.join(str(Path.home()), '.phoner_covid19')
        print("Cache dir: ", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        download_config = datasets.DownloadConfig(cache_dir=cache_dir)
        self._dataset = COVID19(cache_dir=cache_dir)
        print("Cache1 dir:", self._dataset.cache_dir)
        self._dataset.download_and_prepare(download_config=download_config)
        self._dataset = self._dataset.as_dataset()

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self) -> datasets.ClassLabel:
        return self._dataset['train'].features['ner_tags'].feature.names

    @property
    def id2label(self):
        return dict(list(enumerate(self.labels)))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def train(self):
        return self._dataset['train']

    def test(self):
        return self._dataset['test']

    def validation(self):
        return self._dataset['validation']


class RefineCOVID19Dataset(COVID19Dataset):
    NAME = "PhoNER_COVID19"

    def __init__(self):
        cache_dir = os.path.join(str(Path.home()), '.phoner_covid19')
        print("Cache dir: ", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        download_config = datasets.DownloadConfig(cache_dir=cache_dir)
        self._dataset = COVID19(cache_dir=cache_dir,
                                url="https://raw.githubusercontent.com/ducphuE10/NER-Vietnamese-Covid-Dataset/main/train_data/",
                                train_file="train_word_update.conll",
                                )
        print("Cache1 dir:", self._dataset.cache_dir)
        self._dataset.download_and_prepare(download_config=download_config)
        self._dataset = self._dataset.as_dataset()


if __name__ == "__main__":
    dataset = COVID19Dataset().dataset

    print(dataset['train'])
    print(dataset['test'])
    print(dataset['validation'])

    print("List of tags: ",
          dataset['train'].features['ner_tags'].feature.names)

    print("First sample: ", dataset['train'][0])
