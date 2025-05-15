from mj_rag.interfaces import SqlDBServiceInterface
import json
from pathlib import Path


class JsonSqlDBService(SqlDBServiceInterface):

    def __init__(self, folder: str = "sql_db", **kwargs):
        self.folder = Path(folder)
        if not self.folder.exists():
            self.folder.mkdir()

    def add_header_content_in_sdb(self, work_title: str, doc_hash: str, header: str, content: str) -> str:
        json_file = self.folder / f"{doc_hash}.json"
        if json_file.exists():
            with json_file.open() as fp:
                data = json.load(fp)
        else:
            data = {}

        keys = sorted([int(key) for key in data.keys()])
        if keys:
            last_id = keys[-1]
        else:
            last_id = 0

        new_id = last_id + 1
        data[new_id] = {'header': header, 'content': content}
        with json_file.open('w') as fp:
            fp.write(json.dumps(data, indent=2))
        return f"{doc_hash}#{new_id}"

    def get_content_from_id(self, work_title: str, id: str) -> str:
        parts = id.split('#')
        doc_hash, new_id = parts
        new_id = int(new_id)

        json_file = self.folder / f"{doc_hash}.json"
        if json_file.exists():
            with json_file.open() as fp:
                data = json.load(fp)
        else:
            raise FileNotFoundError(f"{json_file} not found")

        if new_id in data:
            return data[new_id]['content']
        elif str(new_id) in data:
            return data[str(new_id)]['content']
        else:
            raise KeyError(f"{new_id} not found in data")
