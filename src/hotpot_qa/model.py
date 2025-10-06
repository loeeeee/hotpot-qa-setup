import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Self, Tuple, TypedDict, Union

from .wikipedia.model import TokenCounter
from .wikipedia.search import WikipediaSearchEngine


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
    distractor_docs: List[Article] = field(default_factory=list)

    @classmethod
    def from_raw_full_wiki(cls, raw_full_wiki: RawFullWikiQA, search_engine: WikipediaSearchEngine) -> Self:
        pk = raw_full_wiki["_id"]
        question = raw_full_wiki["question"]
        answer = raw_full_wiki["answer"]

        supporting_titles = set(title for title, _ in raw_full_wiki["supporting_facts"])
        supporting_docs = []
        for title in supporting_titles:
            results = search_engine.search_title(title)
            if results:
                art = results[0]  # Should be exact match
                supporting_docs.append(Article(title=art.title, text=art.text))
            else:
                logging.warning(f"Supporting article '{title}' not found in Wikipedia dump")

        return cls(pk=pk, question=question, answer=answer, supporting_docs=supporting_docs)

@dataclass
class ArticleQAManager:
    articleqas: List[ArticleQA] = field(default_factory=list)
    token_counter: TokenCounter = field(default_factory=lambda: TokenCounter("gpt"))

    def add(self, qa: ArticleQA) -> None:
        """Add an ArticleQA instance"""
        self.articleqas.append(qa)

    def to_json_per_size(self, data_dir: Path) -> None:
        """
        Save ArticleQA into JSON format for each context size in data_dir
        Skip ArticleQA instances where supporting_docs token count exceeds the context size limit
        """
        import json
        data_dir.mkdir(parents=True, exist_ok=True)

        for size in ContextSize:
            records = []
            for qa in self.articleqas:
                # Calculate total token count of supporting docs
                supporting_texts = "\n".join(doc.text for doc in qa.supporting_docs)
                supporting_token_count = self.token_counter.count_token(supporting_texts)

                # Skip if supporting docs exceed context size limit
                if supporting_token_count > size.value * 1000:
                    logging.warning(f"Skipping QA {qa.pk} for {size.value}k context: supporting docs token count {supporting_token_count} exceeds limit {size.value * 1000}")
                    continue

                # distractors = qa.distractor_docs[size]
                record = {
                    "id": qa.pk,
                    "question": qa.question,
                    "gold_answer": qa.answer,
                    "supporting_docs": [{"title": doc.title, "text": doc.text} for doc in qa.supporting_docs],
                    "distractor_docs": [],
                    "context_size": supporting_token_count
                }
                records.append(record)

            filepath = data_dir / f"long_hotpot_{size.value}k.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
