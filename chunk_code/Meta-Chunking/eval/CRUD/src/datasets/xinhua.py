import json
import os
import random
from typing import Any

from src.datasets.base import BaseDataset

class Xinhua(BaseDataset):
    def __init__(self, data, shuffle: bool = False, seed: int = 22):
        self.data = data

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
            return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]

    def statistics(self) -> dict:
        stat = {'doc': 0, 'gen': 0, 'kno': 0, 'num': 0}
        for type_ in stat.keys():
            stat[type_] = sum([obj['type']==type_ for obj in self.data])
        return stat


def get_task_datasets(path: str, task: str, shuffle: bool = False, seed: int = 22):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"data_path 指向的文件不存在或不是文件: {path}。\n"
            f"请传入 split_merged.json 的绝对路径，例如：\n"
            f"  --data_path C:\\Users\\胡铭强\\Desktop\\chunk_code\\data\\split_merged.json"
        )
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    if task == "all":
        return [
            Xinhua(data["event_summary"], shuffle, seed), 
            Xinhua(data["continuing_writing"], shuffle, seed),
            Xinhua(data["hallu_modified"], shuffle, seed),
            Xinhua(data["questanswer_1doc"], shuffle, seed),
            Xinhua(data["questanswer_2docs"], shuffle, seed),
            Xinhua(data["questanswer_3docs"], shuffle, seed),
        ]
    elif task == "quest_answer":
        def _clean_qa_rows(rows, name):
            clean = [
                row for row in rows
                if isinstance(row.get("questions"), str)
                and row["questions"].strip()
                and isinstance(row.get("answers"), str)
                and row["answers"].strip()
            ]
            dropped = len(rows) - len(clean)
            if dropped:
                print(f"[Dataset] {name}: dropped {dropped} empty QA rows")
            return clean

        return [
            Xinhua(_clean_qa_rows(data["questanswer_1doc"], "questanswer_1doc"), shuffle, seed),
            Xinhua(_clean_qa_rows(data["questanswer_2docs"], "questanswer_2docs"), shuffle, seed),
            Xinhua(_clean_qa_rows(data["questanswer_3docs"], "questanswer_3docs"), shuffle, seed),
        ]
    else:
        return [Xinhua(data[task], shuffle, seed)]
