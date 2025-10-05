from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Self, Tuple, TypedDict, Union


class ContextSize(Enum):
    SM = 8
    MD = 32
    LA = 128


class RawFullWikiQA(TypedDict):
    _id: str
    answer: str
    question: str
    supporting_facts: List[Tuple[str, int]]
    context: List[Tuple[str, List[str]]]
    _type: Literal["comparison", "bridge"]
    level: Literal["hard"]


@dataclass
class Article:
    title: str
    text: str


@dataclass
class ArticleQA:
    pk: str
    question: str
    answer: str
    supporting_docs: List[Article] = field(default_factory=list)
    distractor_docs: Dict[ContextSize, List[Article]] = field(init=False)

    def __post_init__(self) -> None:
        self.distractor_docs = {
            ContextSize.SM: [],
            ContextSize.MD: [],
            ContextSize.LA: [],
        }

    @classmethod
    def from_raw_full_wiki(cls, raw_full_wiki: RawFullWikiQA) -> Self:
        pass
