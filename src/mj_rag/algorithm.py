import json
from typing import List, Tuple, Optional

from pyparsing import originalTextFor, lineStart, nestedExpr

from mj_rag.interfaces import (VectorDBServiceInterface, SqlDBServiceInterface,
                               LoggingServiceInterface, LLMServiceInterface)
import re
from pprint import pformat
import logging
from pathlib import Path
from enum import Enum


class SectionAnswerMode(Enum):
    FIRST_BEST_RAW = "first_best_raw"  # return the "best section" without passing it to llm
    FIRST_BEST_SUMMARY = "first_best_summary"  # ask the llm to resume the "best section"
    TOP_K_RAW = "top_raw"  # return the "best sections" without passing it to llm
    TOP_K_COMBINE = "top_k_combine"  # ask the llm to combine the "best sections"
    TOP_K_SUMMARY = "top_k_summary"  # ask the llm to resume the "best sections"
    TOP_K_RESTRANSCRIPT = "top_k_retranscript"


class MJRagAlgorithm:
    rgx_sentence_limiter = re.compile(r"([\.\?!]+[\n ]+)")
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
        self.llm_service: LLMServiceInterface = llm_service
        self.sql_db_service: SqlDBServiceInterface = sql_db_service or self.get_default_sql_db_service()
        self.add_hierarchized_titles: bool = add_hierachized_titles

    def get_default_logging_service(self) -> LoggingServiceInterface:
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
        from mj_rag.dummy import JsonSqlDBService
        return JsonSqlDBService()

    def save_text_in_databases(self, markdown_content: str):
        self.save_text_as_set_in_vdb(markdown_content)

    def save_text_as_set_in_vdb(self, markdown_content: str, count: int = 5):
        # we make sure the collection for the work exist
        self.vector_db_service.create_collection_for_sentences_set(self.work_title)

        # split the content in sentences
        lines = [senten.strip() for senten in self.rgx_sentence_limiter.split(markdown_content)
                 if senten.strip()]
        for line in lines:
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

        for sentence in sentences_set:
            self.logging_service.debug(f"===> {sentence}")

        # insert the sentences in the the vector db
        self.vector_db_service.insert_sentences_set(self.work_title, sentences_set)

    def save_text_as_titles_in_vdb(self, markdown_content: str):
        # make sure the collection for this work is created
        self.vector_db_service.create_collection_for_section_headers(self.work_title)

        # get the doc's hash and the derived sections
        doc_hash, sections = self.split_content_with_llm(markdown_content)

        # save the section in sql db
        self._save_sections_in_sql_db(doc_hash, sections)
        self.logging_service.debug("After saving in sql db")
        self.logging_service.debug(json.dumps(sections, indent=2))

        # make the sections in row and without subsections
        sections = self._linearize_sections(sections)
        self._remove_subsections_in_sections(sections)
        self.logging_service.debug("After saving in linearization")
        self.logging_service.debug(json.dumps(sections, indent=2))

        # save the sections with their sql doc id in vector database
        self.vector_db_service.insert_section_headers(self.work_title, sections)

    def get_direct_answer(self, question: str, use_alternates: bool = False,
                          use_hypothetical_answers: bool = False) -> str:
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
                "content": 'You are an expert in RAG. You use the context to respond the user question.'
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
                   mode: Optional[SectionAnswerMode] = None):
        classified_answer = self._classify_answer_for_question(question)
        self.logging_service.info(f"{classified_answer = }")

        number_of_sentences = classified_answer['number_of_sentences'].upper()
        kind = classified_answer['kind'].upper() if 'kind' in classified_answer else None

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

    def split_content_with_llm(self, content: str, title: str = None) -> Tuple[str, List[dict]]:
        hash, res = self.get_cached_content_json_tree(content)
        if res:
            return hash, res

        prompt = f"""
The following text is a markdown of a webpage which contains an article. 
{f"The article title is `{title}`." if title else ""}
But this markdown is mixed with many parts of the webpage which are not the actual article or the main content.
You must EXTRACT the actual article or the main content, SPLIT it by markdown headers and RETURN a response in the
following format:

[{{
  "header": "Top level header",
  "content": "El Plan VEA es un programa de ayudas lanzado ...",
  "subsections": [
    {{
      "header": "Mid level header",
      "content": "Los objetivos principales del Plan VEA son...",
      "subsections": [
        {{
          "header": "Low level header",
          "content": "Muchos usuarios temen que las ayudas del ..."
        }}
      ]
    }}
  ]
}}]

Text:
-------
{content}
-------"""

        response = self.llm_service.complete_messages([{'role': 'user', 'content': prompt}])
        resp = self.parse_llm_response_to_json_list(response)
        self.logging_service.debug("Before enriching section")
        self.logging_service.debug(json.dumps(resp, indent=2))
        resp = self.enrich_sections(resp)
        self.logging_service.debug("After enriching section")
        self.logging_service.debug(json.dumps(resp, indent=2))
        self.save_in_cache_content_json_tree(hash, resp)
        return hash, resp

    def generate_summary_from_context_entries(self, context_entries: List[str]) -> str:
        msg_content = f"""Generate a summary of the following context

{self.format_context_entries(context_entries)}"""

        self.logging_service.debug(f"{msg_content = }")

        messages = [
            {"role": "user", "content": msg_content}
        ]
        return self.llm_service.complete_messages(messages)

    def generate_retranscript_from_context_entries(self, context_entries: List[str],
                                                   question: str = None) -> str:
        if question:
            msg_content = f"""Restranscript the following context in a clear and smart way to answer 
this question: {question}

{self.format_context_entries(context_entries)}"""
        else:
            msg_content = f"""Retranscript the following context in a clear and smart way

{self.format_context_entries(context_entries)}"""

        self.logging_service.debug("Message content for Restranscript")
        self.logging_service.debug(msg_content)

        messages = [
            {"role": "user", "content": msg_content}
        ]
        return self.llm_service.complete_messages(messages)

    def combine_context_entries(self, context_entries: List[str], question: str = None) -> str:
        if question:
            msg_content = f"""Combine the following context in a clear and smart way to answer 
this question: {question}

{self.format_context_entries(context_entries)}"""
        else:
            msg_content = f"""Combine the following context in a clear and smart way

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

    def get_cached_content_json_tree(self, content: str) -> Tuple[str, Optional[List[dict]]]:
        cache_dir = Path('content_to_json_cache')
        if not cache_dir.exists():
            cache_dir.mkdir()

        hash = self.get_doc_hash(content)

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

    def enrich_sections(self, sections: list, parents=None, level: int=None) -> list:
        texts = []

        if parents is None:
            parents = []
        if level is None:
            level = len(sections)

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
        return json.loads(results.as_list()[0][0])

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
