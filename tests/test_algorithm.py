from unittest import TestCase
from mj_rag.algorithm import MJRagAlgorithm, SectionAnswerMode
from typing import List
from mj_rag.interfaces import (VectorDBServiceInterface, EmbeddingServiceInterface,
                               SqlDBServiceInterface, LLMServiceInterface)
import os
from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")


class DummyVectorDBService(VectorDBServiceInterface):
    def get_possible_answers_from_question(self, work_title: str, question: str, alternates: List[str] = None,
                                           hypothetical_answers: List[str] = None, top_k: int = 10,
                                           min_score: float = 0.4) -> List[dict]:
        pass

    def get_possible_matchs_from_header(self, work_title: str, sql_db_service: SqlDBServiceInterface, header: str,
                                        alternates: List[str] = None, top_k: int = 10, min_score: float = 0.4) -> List[
        dict]:
        pass

    def insert_section_headers(self, work_title: str, sections: List[dict], **kwargs):
        pass

    def get_embedder_service(self) -> EmbeddingServiceInterface:
        pass

    def create_collection_for_section_headers(self, work_title: str):
        pass

    def create_collection_for_sentences_set(self, work_title: str):
        pass

    def insert_sentences_set(self, work_title: str, sentences_set: List[str]):
        pass


class DummyLLMService(LLMServiceInterface):

    def complete_messages(self, messages: List[dict], **kwargs) -> str:
        return "Dummy response"

with open('texts/flutter-3-29.md') as fp:
    flutter_3_29_text = fp.read()

with open('texts/simple_st.md') as fp:
    simple_st = fp.read()


markdown_content_1 = """Hi. How are you doing?
My name is Jeff. I am 18 years old. And i am someone...

Should You Still Make Use of Docker by 2025?

It depends:
Use Docker if:

    You want a quick, simple interface and are building and testing locally.
    Your staff mostly depends on Docker Compose and traditional approaches.
    You are running straightforward programs without coordination.

Think through substitutes should:

    You are using Kubernetes clusters; a better fit is containerd or CRI-O.
    Particularly in multi-tenant or regulated settings, you need reinforced security.
    You wish to go toward open tools instead of Docker's licensing approach.
    You are maximizing for CI pipelines or macOS performance.

Another popular hybrid strategy is depending on containerd or Podman in CI/CD and production settings while developing locally with Docker.
"""


class MJRagAlgorithmTestCase(TestCase):

    def test_01_splitting_sentences(self):
        algorithm = MJRagAlgorithm("test", DummyVectorDBService(), DummyLLMService())
        algorithm.save_text_in_databases(markdown_content_1)

    def test_02_put_sentences_in_vdb(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("test", MilvusVectorDBService("milvus.db"), DummyLLMService())
        algorithm.save_text_as_set_in_vdb(markdown_content_1, count=3)
        algorithm.vector_db_service.get_possible_answers_from_question("test", "what is the age of Jeff?")
        os.remove('milvus.db')

    def test_03_get_direct_answer(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("test", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_set_in_vdb(markdown_content_1, count=3)
        answer = algorithm.get_direct_answer("what is the age of Jeff?", use_alternates=True)
        print(f"{answer = }")
        os.remove('milvus.db')

    def test_04_get_direct_answer_with_alternates(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("test", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_set_in_vdb(flutter_3_29_text, count=3)
        answer = algorithm.get_direct_answer("what is the behavior when android device doesn't have a vulkan driver?", use_alternates=True)
        print(f"{answer = }")
        os.remove('milvus.db')

    def test_05_get_direct_answer_with_hypothetical_answers(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("test", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_set_in_vdb(flutter_3_29_text, count=3)
        answer = algorithm.get_direct_answer("what is the behavior when android device doesn't have a vulkan driver?",
                                            use_hypothetical_answers=True)
        print(f"{answer = }")
        os.remove('milvus.db')

    def test_06_llm_splitting(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("test", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(simple_st)

    def test_07_vector_db_save(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("test", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(simple_st)
        algorithm.vector_db_service.get_possible_matchs_from_header("test", "logging")
        os.remove('milvus.db')

    def test_08_get_section_as_first_best_raw(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_section_as_answer_from_header("Material", mode=SectionAnswerMode.FIRST_BEST_RAW))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')

    def test_09_get_section_as_first_best_resume(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'o3',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_section_as_answer_from_header("Material", mode=SectionAnswerMode.FIRST_BEST_SUMMARY))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')

    def test_10_get_section_as_top_k_raw(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'o3',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_section_as_answer_from_header("Material", mode=SectionAnswerMode.TOP_K_RAW))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')

    def test_11_get_section_as_top_k_summary(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'o3',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_section_as_answer_from_header("Material", mode=SectionAnswerMode.TOP_K_SUMMARY))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')

    def test_12_get_section_as_top_k_combine(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'o3',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_section_as_answer_from_header("Material", mode=SectionAnswerMode.TOP_K_COMBINE))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')

    def test_13_get_section_with_question_as_first_best_raw(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_section_as_answer_from_question("What is new in flutter 3.29 concerning Material UI?",
                                mode=SectionAnswerMode.FIRST_BEST_RAW))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')

    def test_14_get_section_with_question_as_first_best_summary(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_section_as_answer_from_question("What is new in flutter 3.29 concerning Material UI?",
                                mode=SectionAnswerMode.FIRST_BEST_SUMMARY))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')

    def test_15_get_section_with_question_as_top_k_combine(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_section_as_answer_from_question("What is new in flutter 3.29 concerning Material UI?",
                                mode=SectionAnswerMode.TOP_K_COMBINE))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')

    def test_16_question_classification(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        print(algorithm.get_answer("What is new in flutter 3.29 concerning Material UI?"))

    def test_17_get_answer_for_question(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_answer("What is new in flutter 3.29 concerning Material UI?"))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')

    def test_18_get_answer_for_question(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_set_in_vdb(flutter_3_29_text)
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_answer("Where the property mousecursor has been added?"))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')

    def test_19_get_answer_for_question(self):
        from mj_rag.milvus.vector_db_service import MilvusVectorDBService
        from mj_rag.litellm.llm_service import LiteLLMService

        os.environ.setdefault('OPENAI_API_KEY', OPENAI_API_KEY)
        algorithm = MJRagAlgorithm("Flutter Updates", MilvusVectorDBService("milvus.db"),
                                   LiteLLMService('openai', 'gpt-4o',
                                                  [OPENAI_API_KEY]))
        algorithm.save_text_as_set_in_vdb(flutter_3_29_text)
        algorithm.save_text_as_titles_in_vdb(flutter_3_29_text)
        print(algorithm.get_answer("How to make a toast in flutter 3.29 ?"))
        os.remove('milvus.db')
        os.remove('sql_db/7e0deade99e77bbfb73ae268d6f2ec6ea8c899a8e06f30a762755e48831a9d8f.json')
