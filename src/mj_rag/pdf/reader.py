from math import ceil, floor

import pymupdf
from typing import Union, Optional
import pymupdf4llm
import re
import string


class MjPdfReader:
    rgx_space = re.compile(r" +")

    def __init__(self, file_path: Optional[str] = None,
                 file_content: Optional[Union[str, bytes]] = None):
        if file_path:
            self.doc = pymupdf.open(file_path)
        elif file_content:
            if isinstance(file_content, str):
                file_content = file_content.encode('utf-8')
            self.doc = pymupdf.open(stream=file_content)
        else:
            raise ValueError("Either file_path or file_content must be provided.")

    def get_markdown(self):
        md_text = pymupdf4llm.to_markdown(self.doc)
        recurrent_texts = self.get_list_of_recurrent_texts_as_dict()

        trans_table_d = {}
        for c in ".^$*+-?()[]{}\|":
            trans_table_d[c] = f"\\{c}"

        trans_table = str.maketrans(trans_table_d)

        for signature, value in recurrent_texts.items():
            repeated_set = value['texts']
            for repeated in repeated_set:
                translated = repeated.translate(trans_table)
                repeated_for_rgx = self.rgx_space.sub(" +", translated)

                md_text = re.sub(f"[^\n]*{repeated_for_rgx}[^\n]*", "", md_text)

        md_text = re.sub(r"\n{4,}", "\n\n", md_text)
        return md_text

    def get_list_of_recurrent_texts_as_list(self):
        recurrents = {}
        for page in self.doc:
            text_page = page.get_textpage()

            for block in text_page.extractBLOCKS():
                signature = self.get_block_signature(block)
                recurrents.setdefault(signature, 0)
                recurrents[signature] = recurrents[signature] + 1
        ratio = int(self.doc.page_count * 0.5)
        return [text for text, value in recurrents.items() if value >= ratio]

    def get_list_of_recurrent_texts_as_dict(self):
        recurrents = {}
        for page in self.doc:
            text_page = page.get_textpage()

            for block in text_page.extractBLOCKS():
                signature = self.get_block_signature(block)
                recurrents.setdefault(signature, {'count': 0, 'texts': set()})
                recurrents[signature]['count'] = recurrents[signature]['count'] + 1
                recurrents[signature]['texts'].add(block[4].strip())
        ratio = int(self.doc.page_count * 0.5)

        return {text: value for text, value in recurrents.items() if value['count'] >= ratio}

    def get_block_signature(self, block: tuple):
        x = ceil(block[0])
        y = floor(block[1])
        w = round(block[2])
        h = round(block[3])
        return f"{x}-{y}-{w}-{h}"
