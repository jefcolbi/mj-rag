from unittest import TestCase
from mj_rag.pdf.reader import MjPdfReader
import pymupdf4llm
import re


class TestPdfReader(TestCase):

    def test_get_list_of_recurrent_texts(self):
        reader = MjPdfReader(file_path="texts/simple_pdf.pdf")
        recurrent_texts = reader.get_list_of_recurrent_texts_as_dict(reader.doc)
        print(recurrent_texts)

    def test_O2_to_markdown(self):
        md_text = pymupdf4llm.to_markdown("texts/simple_pdf.pdf",
                                write_images=True, show_progress=True)
        print(md_text)

    def test_03_clean_markdown(self):
        reader = MjPdfReader(file_path="texts/simple_pdf.pdf")
        md_text = reader.get_markdown()
