import sys
from pathlib import Path
# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from vector_store import get_vector_store
vs = get_vector_store('tourist')
vs.delete_collection()
print('已删除tourist向量库')
