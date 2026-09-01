"""
评估指标计算模块
"""
import jieba
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        pass
    
    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        # 清洗文本
        text = text.lower().strip()
        # 使用jieba分词
        tokens = list(jieba.cut(text))
        # 过滤空字符串和标点
        tokens = [t for t in tokens if t.strip() and not t.isspace()]
        return tokens
    
    def calculate_prf(self, pred_tokens: List[str], gt_tokens: List[str]) -> Tuple[float, float, float]:
        """计算Precision, Recall, F1"""
        if not pred_tokens or not gt_tokens:
            return 0.0, 0.0, 0.0
        
        pred_set = set(pred_tokens)
        gt_set = set(gt_tokens)
        
        common = pred_set & gt_set
        
        precision = len(common) / len(pred_set) if pred_set else 0.0
        recall = len(common) / len(gt_set) if gt_set else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return precision, recall, f1
    
    def calculate_bleu(self, reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
        """计算BLEU分数"""
        ref_tokens = self._tokenize(reference)
        hyp_tokens = self._tokenize(hypothesis)
        
        if not hyp_tokens:
            return {
                'bleu_1': 0, 'bleu_2': 0, 'bleu_3': 0, 'bleu_4': 0,
                'bleu_avg': 0.0
            }
        
        bleu_scores = {}
        
        for n in range(1, max_n + 1):
            score = self._bleu_n(ref_tokens, hyp_tokens, n)
            bleu_scores[f'bleu_{n}'] = round(score, 4)
        
        # 计算平均BLEU
        bleu_scores['bleu_avg'] = round(
            np.mean([bleu_scores[f'bleu_{i}'] for i in range(1, max_n + 1)]), 4
        )
        
        return bleu_scores
    
    def _bleu_n(self, ref_tokens: List[str], hyp_tokens: List[str], n: int) -> float:
        """计算BLEU-n"""
        if len(hyp_tokens) < n:
            return 0.0
        
        # 生成n-gram
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        hyp_ngrams = self._get_ngrams(hyp_tokens, n)
        
        if not hyp_ngrams:
            return 0.0
        
        # 计算匹配的n-gram
        ref_counts = Counter(ref_ngrams)
        hyp_counts = Counter(hyp_ngrams)
        
        matches = 0
        for ngram, count in hyp_counts.items():
            matches += min(count, ref_counts.get(ngram, 0))
        
        precision = matches / len(hyp_ngrams) if hyp_ngrams else 0.0
        
        # 简短惩罚
        bp = 1.0
        if len(hyp_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(hyp_tokens)) if len(hyp_tokens) > 0 else 0.0
        
        return bp * precision
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple]:
        """获取n-gram列表"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def calculate_rouge_l(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """计算ROUGE-L分数"""
        ref_tokens = self._tokenize(reference)
        hyp_tokens = self._tokenize(hypothesis)
        
        if not ref_tokens or not hyp_tokens:
            return {'rouge_l_f': 0.0, 'rouge_l_p': 0.0, 'rouge_l_r': 0.0}
        
        lcs_len = self._lcs_length(ref_tokens, hyp_tokens)
        
        precision = lcs_len / len(hyp_tokens) if hyp_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'rouge_l_f': round(f1, 4),
            'rouge_l_p': round(precision, 4),
            'rouge_l_r': round(recall, 4)
        }
