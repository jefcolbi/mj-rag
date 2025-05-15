import re
from typing import List

from mj_rag.interfaces import VectorDBServiceInterface, EmbeddingServiceInterface, SqlDBServiceInterface
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections as milvus_connections,
    utility,
    model
)


class MilvusVectorDBService(VectorDBServiceInterface):

    TEXT_FIELD = "text"
    DENSE_VECTOR_FIELD = "vector"
    ID_FIELD = "id"
    SQL_CONTENT_ID_FIELD = "content_id"
    PARENTS_SEPARATOR = " -> "

    rgx_2_lines = re.compile(r"\n{2,}")

    def __init__(self, uri: str, embedding_dimension: int=768, **kwargs):
        self.uri: str = uri
        self.embedding_dimension: int = embedding_dimension

    def create_collection_for_section_headers(self, work_title: str):
        milvus_connections.connect(uri=self.uri)

        collection_name = self.get_collection_name_for_section_headers(work_title)
        if not utility.has_collection(collection_name):

            fields = [
                FieldSchema(
                    name=self.ID_FIELD,
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=True,
                    max_length=100,
                ),
                FieldSchema(name=self.DENSE_VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dimension),
                FieldSchema(name=self.TEXT_FIELD, dtype=DataType.VARCHAR, max_length=4_096),
                FieldSchema(name=self.SQL_CONTENT_ID_FIELD, dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="level", dtype=DataType.INT8),
                FieldSchema(name="parents", dtype=DataType.VARCHAR, max_length=12_288),
                # FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
            ]

            schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
            collection = Collection(
                name=collection_name, schema=schema, consistency_level="Strong"
            )

            dense_index = {"index_type": "FLAT", "metric_type": "IP"}
            collection.create_index("vector", dense_index)
            # sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            # collection.create_index("sparse_vector", sparse_index)
            collection.flush()
        else:
            collection = Collection(collection_name)

    def create_collection_for_sentences_set(self, work_title: str):
        milvus_connections.connect(uri=self.uri)

        collection_name = self.get_collection_name_for_sentences_set(work_title)
        if not utility.has_collection(collection_name):

            fields = [
                FieldSchema(
                    name=self.ID_FIELD,
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=True,
                    max_length=100,
                ),
                FieldSchema(name=self.DENSE_VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dimension),
                FieldSchema(name=self.TEXT_FIELD, dtype=DataType.VARCHAR, max_length=65_535),
                # FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
            ]

            schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
            collection = Collection(
                name=collection_name, schema=schema, consistency_level="Strong"
            )

            dense_index = {"index_type": "IVF_FLAT", "metric_type": "IP"}
            collection.create_index("vector", dense_index)
            collection.flush()
        else:
            collection = Collection(collection_name)

    def get_possible_answers_from_question(self, work_title: str, question: str,
               alternates: List[str] = None, hypothetical_answers: List[str] = None,
                top_k: int = 10, min_score: float = 0.4) -> List[dict]:
        milvus_connections.connect(uri=self.uri)

        collection_name = self.get_collection_name_for_sentences_set(work_title)
        collection = Collection(collection_name)

        embedder_service = self.get_embedder_service()
        queries = [question]
        if alternates:
            queries.extend(alternates)
        if hypothetical_answers:
            queries.extend(hypothetical_answers)
        query_vectors = embedder_service.encode_queries(queries)

        res = collection.search(query_vectors, self.DENSE_VECTOR_FIELD,
                                {"metric_type": "IP", "params": {"radius": min_score}},
                                top_k, output_fields=[self.TEXT_FIELD])
        answers = []
        for topk_res in res:
            for one_res in topk_res:
                answers.append({'score': one_res['distance'], 'text': one_res['entity'][self.TEXT_FIELD]})
        print(f"{answers = }")
        return answers

    def get_possible_matchs_from_header(self, work_title: str, sql_db_service: SqlDBServiceInterface,
                                        header: str,
                                        alternates: List[str] = None,
                                        top_k:int = 10, min_score: float = 0.4) -> List[dict]:
        milvus_connections.connect(uri=self.uri)

        collection_name = self.get_collection_name_for_section_headers(work_title)
        collection = Collection(collection_name)

        embedder_service = self.get_embedder_service()
        query_vectors = embedder_service.encode_queries([header])

        res = collection.search(query_vectors, self.DENSE_VECTOR_FIELD,
                                {"metric_type": "IP", "params": {"radius": min_score}},
                                top_k, output_fields=[self.TEXT_FIELD, "parents", "level", self.SQL_CONTENT_ID_FIELD])
        answers = []
        content_hashes = []
        for topk_res in res:
            for one_res in topk_res:
                parents = one_res['parents']
                sql_doc_id = one_res['entity'][self.SQL_CONTENT_ID_FIELD]
                content = sql_db_service.get_content_from_id(work_title, sql_doc_id)
                content_hash = self.get_content_hash(content)

                if content_hash in content_hashes:
                    continue
                else:
                    content_hashes.append(content_hash)

                answers.append({
                    'score': one_res['distance'],
                    'header': one_res['entity'][self.TEXT_FIELD],
                    'level': one_res['entity']['level'],
                    'parents': parents.split(self.PARENTS_SEPARATOR) if parents else None,
                    'content': content,
                    'sql_doc_id': sql_doc_id
                })

        if alternates and len(answers) < top_k:
            query_vectors = embedder_service.encode_queries(alternates)

            res = collection.search(query_vectors, self.DENSE_VECTOR_FIELD,
                                    {"metric_type": "IP", "params": {"radius": min_score}},
                                    top_k - len(answers),  # we need to fill only top_k answers
                                    output_fields=[self.TEXT_FIELD, "parents", "level", self.SQL_CONTENT_ID_FIELD])
            for topk_res in res:
                for one_res in topk_res:
                    parents = one_res['parents']
                    sql_doc_id = one_res['entity'][self.SQL_CONTENT_ID_FIELD]
                    content = sql_db_service.get_content_from_id(work_title, sql_doc_id)
                    content_hash = self.get_content_hash(content)

                    if content_hash in content_hashes:
                        continue
                    else:
                        content_hashes.append(content_hash)

                    answers.append({
                        'score': one_res['distance'],
                        'header': one_res['entity'][self.TEXT_FIELD],
                        'level': one_res['entity']['level'],
                        'parents': parents.split(self.PARENTS_SEPARATOR) if parents else None,
                        'content': content,
                        'sql_doc_id': sql_doc_id
                    })

        print(f"{answers = }")
        return answers

    def get_embedder_service(self) -> EmbeddingServiceInterface:
        return model.dense.OpenAIEmbeddingFunction(
                model_name='text-embedding-3-large', # Specify the model name
                dimensions=self.embedding_dimension # Set the embedding dimensionality according to MRL feature.
            )

    def insert_sentences_set(self, work_title: str, sentences_set: List[str]):
        milvus_connections.connect(uri=self.uri)

        collection_name = self.get_collection_name_for_sentences_set(work_title)
        collection = Collection(collection_name)

        embedder_service = self.get_embedder_service()
        vectors = embedder_service.encode_documents(sentences_set)
        data = [
            {self.DENSE_VECTOR_FIELD: vectors[i], self.TEXT_FIELD: sentences_set[i]}
            for i in range(len(vectors))
        ]
        collection.insert(data)

        data.clear()
        vectors.clear()

    def insert_section_headers(self, work_title: str, sections: List[dict], **kwargs):
        milvus_connections.connect(uri=self.uri)

        collection_name = self.get_collection_name_for_section_headers(work_title)
        collection = Collection(collection_name)

        embedder_service = self.get_embedder_service()
        vectors = embedder_service.encode_documents([section['header'] for section in sections])
        data = [
            {
                self.DENSE_VECTOR_FIELD: vectors[i],
                self.TEXT_FIELD: sections[i]['header'],
                self.SQL_CONTENT_ID_FIELD: sections[i]['sql_doc_id'],
                'level': sections[i]['level'],
                'parents': self.PARENTS_SEPARATOR.join(sections[i]['parents'])
            }
            for i in range(len(vectors))
        ]
        collection.insert(data)

        data.clear()
        vectors.clear()
