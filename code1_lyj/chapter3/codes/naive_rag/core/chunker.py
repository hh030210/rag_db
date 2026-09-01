"""
文档切片模块
"""
import re
from typing import List, Dict
import config


class DocumentChunker:
    """文档切片器 - 将长文档切分为小块"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本切分为重叠的块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if not text:
            return []
        
        # 先按句子分割
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 如果当前句子加上已有内容超过块大小，保存当前块
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # 保留重叠部分
                overlap_text = " ".join(current_chunk)
                overlap_sentences = self._split_into_sentences(overlap_text)
                
                # 从后往前找，保留约overlap大小的内容
                current_chunk = []
                current_length = 0
                for s in reversed(overlap_sentences):
                    if current_length + len(s) <= self.chunk_overlap:
                        current_chunk.insert(0, s)
                        current_length += len(s)
                    else:
                        break
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        # 简单的句子分割（按句号、问号、感叹号）
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # 过滤空句子并清理
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        对文档列表进行切片
        
        Args:
            documents: 文档列表，每个文档包含id, question, document等字段
            
        Returns:
            切片后的文档块列表
        """
        chunks = []
        
        for doc in documents:
            doc_id = doc.get('id', '')
            question = doc.get('question', '')
            document_text = doc.get('document', '')
            
            # 对document字段进行切片
            text_chunks = self.split_text(document_text)
            
            for idx, chunk_text in enumerate(text_chunks):
                chunks.append({
                    'doc_id': doc_id,
                    'question': question,
                    'chunk_id': f"{doc_id}_chunk_{idx}",
                    'chunk_index': idx,
                    'text': chunk_text,
                    'metadata': {
                        'original_doc_id': doc_id,
                        'question': question,
                        'chunk_index': idx,
                        'total_chunks': len(text_chunks)
                    }
                })
        
        return chunks


# 全局单例
chunker = DocumentChunker()
