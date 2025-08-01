"""Core implementation of the *MJ‑RAG* retrieval‑augmented‑generation pipeline.

This module contains two public objects:

* ``SectionAnswerMode`` – an :class:`enum.Enum` that lists the different
  post‑processing strategies that can be applied to the raw sections returned
  by the vector database.
* :class:`MJRagAlgorithm` – the high‑level façade class that orchestrates the
  ingestion and querying workflow.  It coordinates four pluggable service
  interfaces (vector DB, SQL/metadata DB, LLM, and logging).

The code follows the *single‑responsibility principle*: each helper method does
exactly one thing (e.g. splitting content, querying the VDB, combining LLM
answers).  Public methods are documented and safe for external use; private
helpers (prefixed with an underscore) are intended for internal use only.

"""
import json
from typing import List, Tuple, Optional, Union, Literal

from pyparsing import originalTextFor, lineStart, nestedExpr

from mj_rag.interfaces import (VectorDBServiceInterface, SqlDBServiceInterface,
                               LoggingServiceInterface, LLMServiceInterface,
                               EmbeddingServiceInterface)
import re
from pprint import pformat
import logging
from pathlib import Path
from enum import Enum
import tiktoken
import numpy as np


class SectionAnswerMode(Enum):
    """Enumeration of high‑level answer‑generation strategies.

    When a user question is mapped to one or more matching *sections* in the
    vector database, we still need to decide *how* to transform those raw
    sections into the final answer string.  ``SectionAnswerMode`` captures the
    available strategies:

    Attributes
    ----------
    FIRST_BEST_RAW
        Return the single best‑scoring section as‑is (no LLM‑post‑processing).
    FIRST_BEST_SUMMARY
        Summarise the single best section with the LLM before returning it.
    TOP_K_RAW
        Return the *k* best sections concatenated, unmodified.
    TOP_K_COMBINE
        Let the LLM combine the *k* best sections into a coherent answer.
    TOP_K_SUMMARY
        Summarise each of the *k* best sections individually and concatenate
        the summaries.
    TOP_K_RESTRANSCRIPT
        Ask the LLM to produce a *retranscript* (rewritten transcript) that
        stitches the *k* best sections together verbatim but in improved prose.
    """
    FIRST_BEST_RAW = "first_best_raw"  # return the "best section" without passing it to llm
    FIRST_BEST_SUMMARY = "first_best_summary"  # ask the llm to resume the "best section"
    TOP_K_RAW = "top_raw"  # return the "best sections" without passing it to llm
    TOP_K_COMBINE = "top_k_combine"  # ask the llm to combine the "best sections"
    TOP_K_SUMMARY = "top_k_summary"  # ask the llm to resume the "best sections"
    TOP_K_RESTRANSCRIPT = "top_k_retranscript"



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class MJRagAlgorithm:
    """High‑level façade that implements the MJ‑RAG workflow.

    The class is **service‑oriented**: external dependencies are injected at
    construction time so that the core logic remains testable and free of
    side‑effects.

    Parameters
    ----------
    work_title : str
        Human‑readable identifier for the *corpus* being processed.  Acts as a
        namespace for the underlying databases.
    vector_db_service : VectorDBServiceInterface
        Service responsible for storage and similarity search over sentence
        embeddings and section headers.
    llm_service : LLMServiceInterface
        Large‑language‑model backend used for classification, summarisation and
        answer generation.
    sql_db_service : SqlDBServiceInterface, optional
        Relational/metadata store used to persist the raw markdown content of
        each section.  A simple JSON‑backed implementation is used by default
        for local development.
    logging_service : LoggingServiceInterface, optional
        Structured logger.  If not provided, :py:meth:`get_default_logging_service`
        is called to create a colourful console logger.
    add_hierachized_titles : bool, default ``True``
        Whether to generate *virtual* section headers that include their parent
        hierarchy (e.g. ``"Parent – Child – Subchild"``) to increase recall
        during header‑based search.
    **kwargs
        Forward‑compatibility hook for subclasses; currently unused.
    """

    rgx_sentence_limiter = re.compile(r"((?:[\.\?!\n][\n ]+)|(?:\n))")
    rgx_only_space = re.compile(r"^[\s\n]*$")
    rgx_md_point = re.compile(r"- (.*)\n")

    rgx_space = re.compile(r" ")
    rgx_2_lines = re.compile(r"\n{2,}")

    def __init__(self, work_title: str,
                 vector_db_service: VectorDBServiceInterface,
                 llm_service: LLMServiceInterface,
                 sql_db_service: SqlDBServiceInterface = None,
                 logging_service: LoggingServiceInterface = None,
                 add_hierachized_titles: bool = True,
                 **kwargs):
        self.work_title: str = work_title
        self.vector_db_service: VectorDBServiceInterface = vector_db_service
        self.logging_service: LoggingServiceInterface = logging_service or self.get_default_logging_service()
        self.embedding_service: EmbeddingServiceInterface = self.vector_db_service.embedding_service
        self.llm_service: LLMServiceInterface = llm_service
        self.sql_db_service: SqlDBServiceInterface = sql_db_service or self.get_default_sql_db_service()
        self.add_hierarchized_titles: bool = add_hierachized_titles

    def get_default_logging_service(self) -> LoggingServiceInterface:
        """Return a stdio when the caller did not supply one."""
        log_format: str = "[%(asctime)s] [%(levelname)s]  %(message)s - %(pathname)s#L%(lineno)s"
        log_date_format: str = "%d/%b/%Y %H:%M:%S"
        console = logging.getLogger(self.work_title)
        console.setLevel(logging.DEBUG)
        hdlr = logging.StreamHandler()
        hdlr.setFormatter(
            logging.Formatter(
                fmt=log_format,
                datefmt=log_date_format,
            )
        )
        hdlr.setLevel(logging.DEBUG)
        console.addHandler(hdlr)
        return console

    def get_default_sql_db_service(self) -> SqlDBServiceInterface:
        """Return a lightweight JSON‑backed metadata store for local use."""
        from mj_rag.dummy import JsonSqlDBService
        return JsonSqlDBService()

    def save_text_in_databases(self, markdown_content: str,
                               source_title: str,
                               source_author: Optional[str] = None,
                               source_url: Optional[str] = None,
                               source_type: Optional[str] = None,
                               doc_hash: Optional[str] = None,
                               ) -> str:
        """Persist *markdown_content* into both the vector and SQL stores."""
        self.save_text_as_set_in_vdb(markdown_content, source_title, source_author=source_author,
                                     source_url=source_url, source_type=source_type,
                                     doc_hash=doc_hash)
        return self.save_text_as_titles_in_vdb(markdown_content, source_title, source_author=source_author,
                                     source_url=source_url, source_type=source_type,
                                     doc_hash=doc_hash)

    def save_text_as_set_in_vdb(self, markdown_content: str,
                                source_title: str,
                                source_author: Optional[str] = None,
                                source_url: Optional[str] = None,
                                source_type: Optional[str] = None,
                                doc_hash: Optional[str] = None,
                                count: int = 5) -> str:
        """Split *markdown_content* into windows of *count* sentences and store them.

        The text is first tokenised by naïvely splitting on punctuation (see
        :pydataattr:`rgx_sentence_limiter`).  Sliding windows of *count*
        sentences (with 1 sentence overlap) are then embedded and up‑serted into the
        *sentence‑set* collection for the current *work_title*.

        Parameters
        ----------
        markdown_content : str
            Raw markdown article.
        source : str
            Source of the markdown content (title of pdf or web page)
        author : str
            Author of markdown content (name of author of the pdf, or url of the site web)
        count : int, default 5
            Window size in sentences.  Each VDB document will contain roughly
            this many sentences.
        """

        # we make sure the collection for the work exist
        self.vector_db_service.create_collection_for_sentences_set(self.work_title)

        doc_hash, sentences_set = self.split_content_by_sentences(markdown_content,
                                                            doc_hash=doc_hash, count=count)
        sentences_vectors = self.get_embeddings_for_sentences(doc_hash, sentences_set)

        # insert the sentences in the vector db
        self.vector_db_service.insert_sentences_set(self.work_title, sentences_set, sentences_vectors,
                                                    source_title, source_author=source_author,
                                                    source_url=source_url, source_type=source_type)
        return doc_hash

    def save_text_as_titles_in_vdb(self, markdown_content: str,
                                   source_title: str,
                                   source_author: Optional[str] = None,
                                   source_url: Optional[str] = None,
                                   source_type: Optional[str] = None,
                                   doc_hash: Optional[str] = None,
                                   ):
        """Extract headed sections from *markdown_content* and index their titles.

        This method performs three steps:

        1. **Split** – use the LLM to identify the hierarchical structure of
           the article and produce a tree of sections (see
           :py:meth:`split_content_with_llm`).
        2. **Persist raw content** – each section (and its parent trace) is
           inserted into the SQL store so we can recover the original
           paragraphs later.
        3. **Persist titles** – finally, we push (header, embedding) pairs to
           the vector DB.  These are later used for header‑based retrieval.
        """

        # make sure the collection for this work is created
        self.vector_db_service.create_collection_for_section_headers(self.work_title)

        # get the doc's hash and the derived sections
        doc_hash, sections = self.split_content_with_llm(markdown_content, doc_hash=doc_hash)

        # save the section in sql db
        self._save_sections_in_sql_db(doc_hash, sections)
        self.logging_service.debug("After saving in sql db")
        self.logging_service.debug(json.dumps(sections, indent=2))

        # make the sections in row and without subsections
        sections = self._linearize_sections(sections)
        if not sections:
            return

        self._remove_subsections_in_sections(sections)
        self.logging_service.debug("After saving in linearization")
        self.logging_service.debug(json.dumps(sections, indent=2))

        # save the sections with their sql doc id in vector database
        self.vector_db_service.insert_section_headers(self.work_title, sections,
                                                      source_title, source_author=source_author,
                                                      source_url=source_url, source_type=source_type
                                                      )
        return doc_hash

    def get_direct_answer(self, question: str, use_alternates: bool = False,
                          use_hypothetical_answers: bool = False) -> str:
        """Return a short direct answer to *question*.

        The method performs an *embedding‑only* lookup against the *sentence‑set*
        collection and passes the returned snippets, together with the user
        question, to the LLM for answer synthesis.

        When *use_alternates* is ``True``, paraphrases of the question are also
        embedded in order to increase recall.  *use_hypothetical_answers*
        injects a handful of synthetic answers as additional query vectors; this
        trick can surface sentences that contain confirming evidence rather than
        re‑phrased questions.
        """
        if use_alternates:
            alternates = self._generate_question_alternates(question)
        else:
            alternates = None

        if use_hypothetical_answers:
            hypothetical_answers = self._generate_hypothetical_answers(question)
        else:
            hypothetical_answers = None

        found_texts = self.vector_db_service.get_possible_answers_from_question(self.work_title, question,
                                    alternates=alternates, hypothetical_answers=hypothetical_answers)
        messages = [
            {
                "role": "system",
                "content": 'You are an expert in RAG. You use the context to respond the user question. Mention your references.'
            },
            {
                "role": "user",
                "content": f"Context:\n\n{json.dumps(found_texts)}"
            },
            {
                "role": "user",
                "content": question
            }
        ]

        answer = self.llm_service.complete_messages(messages)
        return answer

    def get_section_as_answer_from_header(self, section_header: str, use_alternates: bool = True,
                                          mode: SectionAnswerMode = SectionAnswerMode.TOP_K_COMBINE,
                                          known_document_titles: List[str] = None,
                                          top_k: int =5) -> str:
        """Return an answer by searching *section_header* in the *header* collection.

        This helper is used when the caller already knows (or can guess) the
        relevant section title.
        """
        if use_alternates:
            alternates = self._generate_section_alternates(section_header)
            if known_document_titles:
                document_titles: List[str] = known_document_titles
            else:
                document_titles = self._generate_documents_for_section_alternates(section_header)

            header_alternates = []
            for doc_title in document_titles:
                for header in alternates:
                    header_alternates.append(f"{doc_title} - {header}")
        else:
            header_alternates = []

        self.logging_service.debug(f"{header_alternates = }")

        matchs = self.vector_db_service.get_possible_matchs_from_header(self.work_title, self.sql_db_service,
                            section_header, alternates=header_alternates, top_k=top_k)
        return self._process_section_matchs(matchs, mode)

    def get_section_as_answer_from_question(self, question: str, use_alternates: bool = True,
                                          mode: SectionAnswerMode = SectionAnswerMode.TOP_K_COMBINE,
                                          known_document_titles: List[str] = None,
                                          top_k: int =5):
        """Infer probable section headers *from* the question, then delegate to header search."""
        possible_headers = self._generate_possible_headers_from_question(question)
        if use_alternates:
            alternates = self._generate_section_alternates(possible_headers[0])
            if known_document_titles:
                document_titles: List[str] = known_document_titles
            else:
                document_titles = self._generate_documents_for_section_alternates(possible_headers[0])

            for doc_title in document_titles:
                for header in alternates:
                    possible_headers.append(f"{doc_title} - {header}")

        header = possible_headers.pop(0)
        self.logging_service.debug(f"{header = } {possible_headers = }")

        matchs = self.vector_db_service.get_possible_matchs_from_header(self.work_title, self.sql_db_service,
                            header, alternates=possible_headers, top_k=top_k)
        return self._process_section_matchs(matchs, mode, question=question, top_k=top_k)

    def get_answer(self, question:str, top_k: int = 5, return_raw: bool = False,
                   mode: Optional[SectionAnswerMode] = None,
                   number_of_sentences: Union[None, Literal["ONE", "FEW", "TOO_MANY"]] = None):
        """One‑stop shop: decide the best strategy and return an answer to *question*."""
        if number_of_sentences is None:
            classified_answer = self._classify_answer_for_question(question)
            self.logging_service.info(f"{classified_answer = }")

            number_of_sentences = classified_answer['number_of_sentences'].upper()
            kind = classified_answer['kind'].upper() if 'kind' in classified_answer else None
        else:
            if mode == SectionAnswerMode.FIRST_BEST_SUMMARY or mode == SectionAnswerMode.TOP_K_SUMMARY:
                kind = "SUMMARY"
            elif mode == SectionAnswerMode.TOP_K_COMBINE == "COMBINE":
                kind = "COMBINE"
            else:
                kind = None

        if number_of_sentences == "ONE":
            return self.get_direct_answer(question, use_alternates=True,
                                          use_hypothetical_answers=True)
        elif number_of_sentences == "FEW":
            first_answer = self.get_direct_answer(question, use_alternates=True,
                                          use_hypothetical_answers=True)
            self.logging_service.debug(f"{first_answer = }")
            is_good = self.check_if_answer_is_correct(question, first_answer)
            self.logging_service.debug(f"{is_good = }")
            if is_good:
                return first_answer

            if not mode:
                if return_raw:
                    mode = SectionAnswerMode.TOP_K_RAW
                else:
                    mode = SectionAnswerMode.TOP_K_COMBINE
            return self.get_section_as_answer_from_question(question, use_alternates=True,
                            mode=mode, top_k=top_k)
        elif number_of_sentences == "TOO_MANY":
            if not mode:
                if return_raw:
                    mode = SectionAnswerMode.TOP_K_RAW
                elif kind == "SUMMARY":
                    mode = SectionAnswerMode.TOP_K_RESTRANSCRIPT
                else:
                    mode = SectionAnswerMode.TOP_K_COMBINE
            return self.get_section_as_answer_from_question(question, use_alternates=True,
                            mode=mode, top_k=top_k)
        else:
            raise ValueError(f"Wrong value for number_of_sentences '{number_of_sentences}'")

    def get_answer_step_by_step(self, question: str, top_k: int = 5):
        pass

    def check_if_answer_is_correct(self, question: str, answer: str) -> bool:
        """Ask the LLM to sanity‑check *answer* against *question*.

        Returns ``True`` if the LLM replies with *yes*, ``False`` otherwise.
        """

        msg_content = f"""We are working on a document which the document is turning around '{self.work_title}'

The user asked: {question}

The assistant fetched the answer in the document and replied with: {answer}

Is the assitant's answer looks like a good answer? Reply by YES or NO"""

        messages = [
            {"role": "user", "content": msg_content}
        ]
        to_parse = self.llm_service.complete_messages(messages)
        if 'yes' in to_parse.lower():
            return True
        else:
            return False

    def _classify_answer_for_question(self, question: str) -> dict:
        msg_content = f"""The user is asking this question: "{question}".
The answers to this question is inside a vector database. Your goal is to help us find these informations.
        
Based on user's question you must guess if the answer can fit in ONE sentence, FEW sentences or TOO MANY sentences.
You must also guess which kind of answer will be the best for the user when the answer will be in 
TOO MANY sentences: a SUMMARY of results found or a COMBINATION of these results.

---------------------------------------

Let me show you some examples:

^^^^^^^^^^^^^^^^^^^^^^^^^^
Question: What is the birth date of Donald Trump Junior
Your answer: {{"reasoning": "The user is asking for a birth date which can be replied in one sentence", "number_of_sentences": "ONE"}}

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Question: When did the conflict end?
Your answer: {{"reasoning": "The user is asking for ...", "number_of_sentences": "ONE"}}

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Question: How to cook a pizza?
Your answer: {{"reasoning": "The user is asking for ...", "number_of_sentences": "FEW"}}

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Question: What were the causes of Matthew departure?
Your answer: {{"reasoning": "The user is asking for ...", "number_of_sentences": "FEW"}}

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Question: Tell me everything you can find about Rust weaknesses
Your answer: {{"reasoning": "The user is asking for ... and we must combine all the results", 
"number_of_sentences": "TOO_MANY", "kind": "COMBINING"}}

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Question: What can you tell me about Rust?
Your answer: {{"reasoning": "The user is asking for a summary of everything we can find about Rust", 
"number_of_sentences": "FEW", "kind": "SUMMARY"}}

---------------------------------------

{question}

Your answer: """

        messages = [
            {"role": "user", "content": msg_content}
        ]
        to_parse = self.llm_service.complete_messages(messages)
        self.logging_service.debug(to_parse)
        return self._extract_to_json_object(to_parse)

    def _process_section_matchs(self, matchs: List[dict], mode: SectionAnswerMode,
                                top_k:int=5, question: str = None) -> str:
        if mode == SectionAnswerMode.FIRST_BEST_RAW:
            return self._get_content_from_sql_db_from_id(matchs[0]['sql_doc_id'])
        elif mode == SectionAnswerMode.FIRST_BEST_SUMMARY:
            return self.generate_summary_from_context_entries(
                [self._section_match_to_context_entry(matchs[0])]
            )
        elif mode == SectionAnswerMode.TOP_K_RAW:
            return self.format_context_entries(
                [self._section_match_to_context_entry(match) for match in matchs[:top_k]]
            )
        elif mode == SectionAnswerMode.TOP_K_SUMMARY:
            return self.generate_summary_from_context_entries(
                [self._section_match_to_context_entry(match) for match in matchs[:top_k]]
            )
        elif mode == SectionAnswerMode.TOP_K_COMBINE:
            return self.combine_context_entries(
                [self._section_match_to_context_entry(match) for match in matchs[:top_k]],
                question=question
            )
        elif mode == SectionAnswerMode.TOP_K_RESTRANSCRIPT:
            return self.generate_retranscript_from_context_entries(
                [self._section_match_to_context_entry(match) for match in matchs[:top_k]],
                question=question
            )

    def _section_match_to_context_entry(self, section_match: dict) -> str:
        header: str = section_match['header']
        parents: List[str] = section_match['parents']
        if parents:
            header = header.replace(' - '.join(parents), '').strip()
            header = header.lstrip('-').strip()
            parents_hierarchy = " -> ".join(parents)
        else:
            parents_hierarchy = ""
        content = f"Header: {header}\nParents Hierarchy: {parents_hierarchy}"
        content += f"\nLevel: {section_match['level']}\nSemantic score: {section_match['score']}"
        content += f"\nSource title: {section_match['source_title']}"
        if section_match.get('source_url'):
            content += f"\nSource url: {section_match['source_url']}"
        if section_match.get('source_author'):
            content += f"\nSource author: {section_match['source_author']}"
        if section_match.get('source_type'):
            content += f"\nSource type: {section_match['source_type']}"
        content += f"\nContent: {section_match['content']}"
        return content

    def _get_content_from_sql_db_from_id(self, doc_id: str) -> str:
        return self.sql_db_service.get_content_from_id(self.work_title, doc_id)

    def _generate_section_alternates(self, section_header: str) -> List[str]:
        msg_content = f"""We are working on a document which the document is turning around '{self.work_title}'
        
Inside this document there is a section which header is: {section_header}
Give us few alternate section headers that mean the same thing as '{section_header}'

Answer with the following format:

---------------------
- alternative header 1
- alternative header 2
---------------------"""

        messages = [
            {"role": "user", "content": msg_content}
        ]
        to_parse = self.llm_service.complete_messages(messages)
        headers = self._extract_points(to_parse)
        self.logging_service.debug(f"{headers}")
        return headers

    def _generate_documents_for_section_alternates(self, section_header: str) -> List[str]:
        msg_content = f"""We are working on a subject wich turns around '{self.work_title}'

We ask you to give us some SHORT document titles which subject turn around '{self.work_title}' 
and which contains a section which header is {section_header}.
These documents are broader and not specific to {section_header}.
{section_header} is just a section in these documents.

Answer with the following format:

---------------------
- alternative header 1
- alternative header 2
---------------------"""

        messages = [
            {"role": "user", "content": msg_content}
        ]
        to_parse = self.llm_service.complete_messages(messages)
        doc_titles = self._extract_points(to_parse)
        doc_titles = [doc_title.split(':')[0].strip() for doc_title in doc_titles]
        self.logging_service.debug(f"{doc_titles}")
        return doc_titles

    def _generate_question_alternates(self, question: str) -> List[str]:
        msg_content = f"""Generate few alternative questions with the same meaning 
for this question: {question}

Answer with the following format:

---------------------
- alternative question 1
- alternative question 2
---------------------"""

        messages = [
            {"role": "user", "content": msg_content}
        ]
        to_parse = self.llm_service.complete_messages(messages)
        points = self._extract_points(to_parse)
        self.logging_service.debug(f"{points}")
        return points

    def _generate_hypothetical_answers(self, question: str) -> List[str]:
        """Generate synthetic answers that *could* answer *question*.

        This scoring‑trick sometimes surfaces paragraphs that contain facts but
        not the exact phrasing of the query.
        """

        msg_content = f"""Generate few hypothetical answers with the same meaning 
for this question: {question}

Answer with the following format:

---------------------
- xxx yyy zzz...
- mmm nnn ooo...
---------------------"""

        messages = [
            {"role": "user", "content": msg_content}
        ]
        to_parse = self.llm_service.complete_messages(messages)
        points = self._extract_points(to_parse)
        self.logging_service.debug(f"{points}")
        return points

    def _generate_possible_headers_from_question(self, question: str) -> List[str]:
        msg_content = f""""{question}" is the question of a user.
The answer to this question is a section inside a document. We don't know 
that section's header. Give us few section's headers which we will use to do 
search in a vector database. Your generated headers must be SHORT.

Answer with the following format:

---------------------
- alternative question 1
- alternative question 2
---------------------"""

        messages = [
            {"role": "user", "content": msg_content}
        ]
        to_parse = self.llm_service.complete_messages(messages)
        points = self._extract_points(to_parse)
        self.logging_service.debug(f"{points}")
        return points

    def _extract_points(self, content: str) -> List[str]:
        res = [point.strip() for point in self.rgx_md_point.findall(content)]
        res = [point for point in res if point]
        return res

    def _extract_to_json_object(self, response: str):
        nester_expr = originalTextFor(lineStart + nestedExpr("{", "}"))
        results = nester_expr.search_string(response)
        return json.loads(results.as_list()[0][0])

    def split_content_with_llm(self, content: str, title: str = None,
                               doc_hash: Optional[str] = None) -> Tuple[str, List[dict]]:
        """Ask the LLM to parse *content* into a *header/content* JSON tree.

        Results are cached on disk (see :py:meth:`get_cached_content_json_tree`) so
        that repeated ingestion of the same document does not incur extra token
        costs.
        """
        hash, res = self.get_cached_content_json_tree(content, doc_hash=doc_hash)
        if res:
            return hash, res

        if not content:
            raise ValueError("Empty content")

        prompt = f"""
The following text is a markdown of a webpage which contains an article. 
{f"The article title is `{title}`." if title else ""}

Rules:
1. This markdown is mixed with many parts of the webpage which are not the actual article or the main content.
2. Analyze carefully the markdown and assume there can be some errors in heading and spacing.
3. You must EXTRACT the actual article or the main content, ORGANIZE it by markdown headers, 
spacing and semantic.
4. If you can't identify a header consider it is the same content.
5. Recopy the content of each header AS IS. SUMMARIZATION NOT ALLOWED. Just fix punctuation and spacing issues.
6. Your response will be parsed. Avoid any comment or json errors.
7. If there is no header present, propose your own header based on the content
8. RETURN a response in the following format:

[{{
  "header": "Top level header",
  "content": "El Plan VEA es un programa de ayudas lanzado ... until the end",
  "subsections": [
    {{
      "header": "Mid level header",
      "content": "Los objetivos principales del Plan VEA son ... until the end",
      "subsections": [
        {{
          "header": "Low level header",
          "content": "Muchos usuarios temen que las ayudas del ... until the end"
        }}
      ]
    }}
  ]
}}]

Text:
-------
{content}
-------"""

        encoding = tiktoken.encoding_for_model(self.embedding_service.model_name)
        tokens_count = len(encoding.encode(prompt))
        if tokens_count > 100000:
            raise ValueError(f"TOO MANY TOKENS {tokens_count = }")

        errors = []
        for _ in range(5):
            try:
                messages = [{'role': 'user', 'content': prompt}]
                messages.extend(errors)
                response = self.llm_service.complete_messages(messages)
                resp = self.parse_llm_response_to_json_list(response)
                break
            except json.JSONDecodeError as e:
                errors.append({'role': 'user', 'content': f"Your response is not valid JSON: {e}"})
        else:
            raise ValueError("The LLM can't return valid JSON after 5 tries.")

        self.logging_service.debug("Before enriching section")
        self.logging_service.debug(json.dumps(resp, indent=2))
        resp = self.enrich_sections(resp)
        self.logging_service.debug("After enriching section")
        self.logging_service.debug(json.dumps(resp, indent=2))
        self.save_in_cache_content_json_tree(hash, resp)
        return hash, resp

    def split_content_by_sentences(self, markdown_content: str, doc_hash: str = None,
                                   count: int = 5):
        hash, res = self.get_cached_content_sentences(markdown_content, doc_hash=doc_hash)
        if res:
            return hash, res

        if not markdown_content:
            raise ValueError("Empty content")

        encoding = tiktoken.encoding_for_model(self.embedding_service.model_name)

        # split the content in sentences
        lines = [senten.strip() for senten in self.rgx_sentence_limiter.split(markdown_content)
                 if senten.strip()]
        for line in lines:
            # tokens_count = len(encoding.encode(line))
            self.logging_service.debug(line)

        # build the sentences set
        total_lines = len(lines)
        sentences_set = []
        for i in range(0, total_lines - 3, 2):
            step = i + (count * 2)
            sentence = ""
            for j, part in enumerate(lines[i:step]):
                if self.rgx_sentence_limiter.match(part):
                    sentence = f"{sentence}{part}"
                else:
                    sentence = f"{sentence} {part}"
            sentences_set.append(sentence)

        max_tokens_count = 0
        for sentence in sentences_set:
            tokens_count = len(encoding.encode(sentence))
            max_tokens_count = max(max_tokens_count, tokens_count)
            self.logging_service.debug(f"===> {tokens_count} {sentence}")

        self.logging_service.info(f"{max_tokens_count = }")
        self.save_in_cache_content_sentences(hash, sentences_set)
        return hash, sentences_set

    def generate_summary_from_context_entries(self, context_entries: List[str]) -> str:
        msg_content = f"""Generate a summary of the following context and cite your sources

{self.format_context_entries(context_entries)}"""

        self.logging_service.debug(f"{msg_content = }")

        messages = [
            {"role": "user", "content": msg_content}
        ]
        return self.llm_service.complete_messages(messages)

    def generate_retranscript_from_context_entries(self, context_entries: List[str],
                                                   question: str = None) -> str:
        if question:
            msg_content = f"""Restranscript the following context in a clear and smart way, and cite your sources to answer
this question: {question}

{self.format_context_entries(context_entries)}"""
        else:
            msg_content = f"""Retranscript the following context in a clear and smart way, and cite your sources

{self.format_context_entries(context_entries)}"""

        self.logging_service.debug("Message content for Restranscript")
        self.logging_service.debug(msg_content)

        messages = [
            {"role": "user", "content": msg_content}
        ]
        return self.llm_service.complete_messages(messages)

    def combine_context_entries(self, context_entries: List[str], question: str = None) -> str:
        if question:
            msg_content = f"""Combine the following context in a clear and smart way, and cite your sources to answer 
this question: {question}

{self.format_context_entries(context_entries)}"""
        else:
            msg_content = f"""Combine the following context in a clear and smart way, and cite your sources

{self.format_context_entries(context_entries)}"""

        self.logging_service.debug("Message content for Combine")
        self.logging_service.debug(msg_content)

        messages = [
            {"role": "user", "content": msg_content}
        ]
        return self.llm_service.complete_messages(messages)

    def format_context_entries(self, context_entries: List[str]) -> str:
        ctx = '\n~~~~~~~~~\n'.join(context_entries)
        return f"++++++++++++++++\n{ctx}\n++++++++++++++++"

    def get_cached_content_json_tree(self, content: str, doc_hash: Optional[str] = None) -> Tuple[str, Optional[List[dict]]]:
        cache_dir = Path('content_to_json_cache')
        if not cache_dir.exists():
            cache_dir.mkdir()

        hash = self.get_doc_hash(content) if not doc_hash else doc_hash

        self.logging_service.debug(f"{hash = }")

        cache_file = cache_dir / f"{hash}.json"
        if cache_file.exists():
            with cache_file.open() as fp:
                res = json.load(fp)
                return hash, res
        else:
            return hash, None

    def get_cached_content_embeddings(self, content: Optional[str] = None,
                                      doc_hash: Optional[str] = None)  -> Tuple[str, Optional[List[dict]]]:
        cache_dir = Path('content_to_vector_cache')
        if not cache_dir.exists():
            cache_dir.mkdir()

        if not doc_hash:
            hash = self.get_doc_hash(content)
        else:
            hash = doc_hash

        self.logging_service.debug(f"{hash = }")

        cache_file = cache_dir / f"{hash}.npy"
        if cache_file.exists():
            res = np.load(str(cache_file))
            return hash, res
        else:
            return hash, None

    def get_cached_content_sentences(self, content: str, doc_hash: Optional[str] = None) -> Tuple[str, Optional[List[dict]]]:
        cache_dir = Path('content_to_sentence_cache')
        if not cache_dir.exists():
            cache_dir.mkdir()

        hash = self.get_doc_hash(content) if not doc_hash else doc_hash

        self.logging_service.debug(f"{hash = }")

        cache_file = cache_dir / f"{hash}.json"
        if cache_file.exists():
            with cache_file.open() as fp:
                res = json.load(fp)
                return hash, res
        else:
            return hash, None

    def get_doc_hash(self, content: str) -> str:
        from hashlib import sha256
        content = self.rgx_space.sub('', content)
        content = self.rgx_2_lines.sub('\n', content)

        m = sha256(content.encode())
        return m.hexdigest()

    def save_in_cache_content_json_tree(self, hash: str, json_tree: List[dict]):
        cache_dir = Path('content_to_json_cache')
        if not cache_dir.exists():
            cache_dir.mkdir()

        cache_file = cache_dir / f"{hash}.json"
        with cache_file.open('w') as fp:
            fp.write(json.dumps(json_tree, indent=2))

    def save_in_cache_content_embeddings(self, hash: str, embeddings: np.ndarray):
        cache_dir = Path('content_to_vector_cache')
        if not cache_dir.exists():
            cache_dir.mkdir()

        cache_file = cache_dir / f"{hash}.npy"
        np.save(str(cache_file), embeddings)

    def save_in_cache_content_sentences(self, hash: str, sentences_set: List[str]):
        cache_dir = Path('content_to_sentence_cache')
        if not cache_dir.exists():
            cache_dir.mkdir()

        cache_file = cache_dir / f"{hash}.json"
        with cache_file.open('w') as fp:
            fp.write(json.dumps(sentences_set, indent=2))

    def get_embeddings_for_sentences(self, doc_hash: str, sentences_set: List[str],
                                     count: int = 5):
        _, res = self.get_cached_content_embeddings(doc_hash=doc_hash)
        if res is not None:
            return res

        encoding = tiktoken.encoding_for_model(self.embedding_service.model_name)
        vectors = []
        tmp_sentences = []
        tmp_tokens_count = 0
        for i in range(len(sentences_set)):

            sentences = sentences_set[i]
            tokens_count = len(encoding.encode(sentences))

            if tmp_tokens_count + tokens_count > 250000:
                try:
                    vectors.extend(self.embedding_service.encode_documents(tmp_sentences))
                    tmp_sentences.clear()
                    tmp_sentences.append(sentences)
                    tmp_tokens_count = tokens_count
                except Exception as e:
                    if 'maximum context length is 8192' in str(e):
                        # Find a way to split the text into smaller chunks and encode them separately
                        pass
                    raise e
            else:
                tmp_sentences.append(sentences)
                tmp_tokens_count += tokens_count

        if tmp_sentences:
            try:
                vectors.extend(self.embedding_service.encode_documents(tmp_sentences))
            except Exception as e:
                if 'maximum context length is 8192' in str(e):
                    # Find a way to split the text into smaller chunks and encode them separately
                    pass
                raise e

        self.save_in_cache_content_embeddings(doc_hash, vectors)
        return vectors


    def enrich_sections(self, sections: list, parents=None, level: int=None) -> list:
        texts = []

        if parents is None:
            parents = []
            level = 1
        # if level is None:
        #     level = len(sections)

        for section in sections:
            header = section['header']
            content = f"{'#'*level} {header}\n\n{section['content']}"

            subsections = section.get('subsections')
            if subsections:
                parents_to_pass = parents.copy()
                parents_to_pass.append(header)
                section['subsections'] = self.enrich_sections(subsections,
                                                              parents=parents_to_pass, level=level+1)

                for subsection in section['subsections']:
                    content += f"\n\n{subsection['content']}"

            section['content'] = content
            section['parents'] = parents
            section['level'] = level
            texts.append(section)

        return texts

    def parse_llm_response_to_json_list(self, response: str) -> List[dict]:
        nester_expr = originalTextFor(lineStart + nestedExpr("[", "]"))
        results = nester_expr.search_string(response)
        res_json_str = results.as_list()[0][0]
        return json.loads(res_json_str)

    def parse_llm_response_to_json_object(self, response: str) -> List[dict]:
        nester_expr = originalTextFor(lineStart + nestedExpr("{", "}"))
        results = nester_expr.search_string(response)
        res_json_str = results.as_list()[0][0]
        return json.loads(res_json_str)

    def _save_sections_in_sql_db(self, doc_hash: str, sections: List[dict]):
        for i, section in enumerate(sections):
            sql_doc_id = self.sql_db_service.add_header_content_in_sdb(
                self.work_title, doc_hash, section['header'], section['content']
            )
            sections[i]['sql_doc_id'] = sql_doc_id

            if 'subsections' in section and section['subsections']:
                self._save_sections_in_sql_db(doc_hash, sections[i]['subsections'])

    def _linearize_sections(self, sections: List[dict]) -> List[dict]:
        res = []
        for section in sections:
            res.append(section)

            if self.add_hierarchized_titles:
                # reverse_parents = list(reversed(section.get('parents', [])))
                if section.get('parents', []):
                    parents = section.get('parents', [])
                    for i in range(len(parents) - 1, -1, -1):
                        to_include = parents[i:]
                        to_include.append(section['header'])
                        new_section = section.copy()
                        new_section['header'] = " - ".join(to_include)
                        res.append(new_section)

            res.extend(self._linearize_sections(section.get('subsections', [])))

        return res

    def _remove_subsections_in_sections(self, sections: List[dict]):
        for i in range(len(sections)):
            sections[i].pop('subsections', None)
            sections[i].pop('content', None)

    def fix_header_in_section(self, title: str, section: dict):
        msg_content = f""""I have an article with title {title}.

One of its section has been parsed to json like this

{json.dumps(section, indent=2)}

But as you can see the header is missing. Modify that json by setting the header based on 
the content and return me the modified json."""

        messages = [
            {"role": "user", "content": msg_content}
        ]
        response = self.llm_service.complete_messages(messages)
        resp = self.parse_llm_response_to_json_object(response)
