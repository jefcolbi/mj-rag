from typing import List, Protocol, Optional
import re


class EmbeddingServiceInterface(Protocol):
    dimensions: int
    model_name: str

    def encode_documents(self, documents: List[str]) -> List[List[List[float]]]:
        raise NotImplementedError

    def encode_queries(self, queries: List[str]) -> List[List[List[float]]]:
        raise NotImplementedError


class SqlDBServiceInterface(Protocol):

    def add_header_content_in_sdb(self, work_title: str, doc_hash: str,
                    header: str, content: str) -> str:
        raise NotImplementedError

    def get_content_from_id(self, work_title: str, id: str) -> str:
        raise NotImplementedError


class VectorDBServiceInterface(Protocol):
    rgx_space = re.compile(r"\s+")
    rgx_2_lines = re.compile(r"\n{2,}")
    embedding_service: EmbeddingServiceInterface

    def get_collection_name_for_sentences_set(self, work_title: str):
        canon = self.rgx_space.sub('_', work_title)
        return f"collection_sentences_{canon}"

    def get_collection_name_for_section_headers(self, work_title: str):
        canon = self.rgx_space.sub('_', work_title)
        return f"collection_sections_{canon}"

    def create_collection_for_section_headers(self, work_title: str):
        raise NotImplementedError

    def create_collection_for_sentences_set(self, work_title: str):
        raise NotImplementedError

    def get_possible_answers_from_question(self, work_title: str, question: str,
               alternates: List[str]=None, hypothetical_answers: List[str]=None,
                            top_k:int = 10, min_score: float = 0.4) -> List[dict]:
        raise NotImplementedError

    def get_possible_matchs_from_header(self, work_title: str, sql_db_service: SqlDBServiceInterface,
                                        header: str,
                                        alternates: List[str] = None,
                                        top_k:int = 10, min_score: float = 0.4) -> List[dict]:
        raise NotImplementedError

    def insert_sentences_set(self, work_title:str, sentences_set: List[str],
                             sentences_vectors: List[List[List[float]]],
                             source_title: str,
                             source_author: Optional[str] = None,
                             source_url: Optional[str] = None,
                             source_type: Optional[str] = None,
                             ):
        raise NotImplementedError

    def insert_section_headers(self, work_title:str, sections: List[dict],
                               source_title: str,
                               source_author: Optional[str] = None,
                               source_url: Optional[str] = None,
                               source_type: Optional[str] = None,
                               **kwargs):
        """

        :param work_title:
        :param sections: list of section-dictionaries
        :param source_title:
        :param source_author:
        :param source_url:
        :param source_type:
        :param kwargs:
        """
        raise NotImplementedError

    def get_content_hash(self, content: str) -> str:
        from hashlib import sha256
        content = self.rgx_space.sub('', content)
        content = self.rgx_2_lines.sub('\n', content)

        m = sha256(content.encode())
        return m.hexdigest()


class LoggingServiceInterface(Protocol):

    def debug(self, message: str, **kwargs):
        raise NotImplementedError

    def info(self, message: str, **kwargs):
        raise NotImplementedError

    def warning(self, message: str, **kwargs):
        raise NotImplementedError

    def error(self, message: str, **kwargs):
        raise NotImplementedError


class LLMServiceInterface(Protocol):
    def complete_messages(self, messages: List[dict], **kwargs) -> str:
        raise NotImplementedError
