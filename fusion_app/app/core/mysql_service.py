"""
MySQL 数据库服务

管理 MySQL 连接，提供：
- 基础 CRUD 操作
- MainIndex 表操作
- 提示优化相关表操作（prompt_clusters, prompt_iterations, tourist_questions）
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from config import get_fusion_config
    _cfg = get_fusion_config()
except Exception:
    _cfg = None


class MySQLService:
    """MySQL 服务封装"""

    _instance: Optional["MySQLService"] = None

    def __init__(self, config=None):
        self.cfg = config or _cfg
        self._conn = None
        self._cursor = None

    # ==================== 连接管理 ====================

    def connect(self) -> bool:
        """建立 MySQL 连接"""
        if self._conn and self._is_connected():
            return True

        try:
            import mysql.connector
            from mysql.connector import Error

            rdb_cfg = self.cfg.rdb if self.cfg else None
            if rdb_cfg is None:
                print("[MySQL] 配置未找到")
                return False

            self._conn = mysql.connector.connect(
                host=rdb_cfg.host,
                port=rdb_cfg.port,
                user=rdb_cfg.user,
                password=rdb_cfg.password,
                charset="utf8mb4",
                autocommit=False,
            )
            self._cursor = self._conn.cursor(dictionary=True)
            print(f"[MySQL] 连接成功: {rdb_cfg.host}:{rdb_cfg.port}")
            return True

        except ImportError:
            print("[MySQL] mysql-connector-python 未安装")
            return False
        except Exception as e:
            print(f"[MySQL] 连接失败: {e}")
            return False

    def _is_connected(self) -> bool:
        try:
            if self._conn:
                self._conn.ping(reconnect=False)
                return True
        except Exception:
            pass
        return False

    def ensure_connection(self) -> bool:
        """确保连接有效，不行则重连"""
        if not self._is_connected():
            self._cursor = None
            self._conn = None
            return self.connect()
        return True

    def close(self):
        """关闭连接"""
        if self._cursor:
            try:
                self._cursor.close()
            except Exception:
                pass
            self._cursor = None
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def commit(self):
        """提交事务"""
        if self._conn:
            self._conn.commit()

    def rollback(self):
        """回滚事务"""
        if self._conn:
            self._conn.rollback()

    def execute(self, sql: str, params: tuple = None) -> Any:
        """执行 SQL"""
        if not self.ensure_connection():
            raise RuntimeError("MySQL 未连接")
        try:
            self._cursor.execute(sql, params or ())
            return self._cursor.fetchall()
        except Exception as e:
            self.rollback()
            raise e

    def execute_one(self, sql: str, params: tuple = None) -> Optional[Dict]:
        """执行 SQL，返回一条"""
        if not self.ensure_connection():
            raise RuntimeError("MySQL 未连接")
        try:
            self._cursor.execute(sql, params or ())
            return self._cursor.fetchone()
        except Exception as e:
            self.rollback()
            raise e

    def execute_write(self, sql: str, params: tuple = None) -> int:
        """执行写入 SQL，返回影响行数"""
        if not self.ensure_connection():
            raise RuntimeError("MySQL 未连接")
        try:
            self._cursor.execute(sql, params or ())
            self.commit()
            return self._cursor.rowcount
        except Exception as e:
            self.rollback()
            raise e

    def execute_many(self, sql: str, params_list: List[tuple]) -> int:
        """批量执行写入"""
        if not self.ensure_connection():
            raise RuntimeError("MySQL 未连接")
        try:
            self._cursor.executemany(sql, params_list)
            self.commit()
            return self._cursor.rowcount
        except Exception as e:
            self.rollback()
            raise e

    # ==================== 数据库/表操作 ====================

    def create_database_if_not_exists(self, database: str):
        """创建数据库"""
        if not self.ensure_connection():
            return False
        try:
            self.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            self.execute(f"USE `{database}`")
            print(f"[MySQL] 数据库 {database} 就绪")
            return True
        except Exception as e:
            print(f"[MySQL] 创建数据库失败: {e}")
            return False

    def use_database(self, database: str):
        """切换数据库"""
        if not self.ensure_connection():
            return False
        self.execute(f"USE `{database}`")
        return True

    def show_tables(self) -> List[str]:
        """列出所有表"""
        rows = self.execute("SHOW TABLES")
        if not rows:
            return []
        key = list(rows[0].keys())[0]
        return [list(r.values())[0] for r in rows]

    def table_exists(self, table: str) -> bool:
        """检查表是否存在"""
        try:
            rows = self.execute("SHOW TABLES LIKE %s", (table,))
            return len(rows) > 0
        except Exception:
            return False

    def describe_table(self, table: str) -> List[Dict]:
        """获取表结构"""
        return self.execute(f"DESCRIBE `{table}`")

    # ==================== MainIndex 表操作（RAG_DB_slim）====================

    def init_mainindex(self, table: str = None, force: bool = False) -> Dict[str, Any]:
        """初始化 MainIndex 表"""
        if not self.ensure_connection():
            return {"status": "error", "message": "MySQL 未连接"}

        rdb_cfg = self.cfg.rdb
        db = rdb_cfg.database
        tbl = table or rdb_cfg.table

        self.create_database_if_not_exists(db)
        self.use_database(db)

        if self.table_exists(tbl):
            if force:
                self.execute(f"DROP TABLE `{tbl}`")
                print(f"[MySQL] 已删除旧表: {tbl}")
            else:
                return {"status": "exists", "message": f"表 {tbl} 已存在"}

        sql = f"""
        CREATE TABLE `{tbl}` (
            `doc_id` VARCHAR(255) PRIMARY KEY,
            `corpus_id` VARCHAR(255),
            `doc_text` LONGTEXT,
            `profile_json` JSON,
            `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX `idx_corpus_id` (`corpus_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        self.execute(sql)
        print(f"[MySQL] 创建表: {db}.{tbl}")
        return {"status": "created", "message": f"表 {tbl} 创建成功"}

    def insert_mainindex(self, rows: List[Dict]) -> int:
        """批量写入 MainIndex"""
        if not rows:
            return 0

        rdb_cfg = self.cfg.rdb
        tbl = rdb_cfg.table
        self.use_database(rdb_cfg.database)

        sql = f"""
        INSERT INTO `{tbl}` (doc_id, corpus_id, doc_text, profile_json, updated_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            corpus_id = VALUES(corpus_id),
            doc_text = VALUES(doc_text),
            profile_json = VALUES(profile_json)
        """

        params = []
        for row in rows:
            profile = json.dumps(row.get("profile_json", {}), ensure_ascii=False)
            params.append((
                row["doc_id"],
                row.get("corpus_id", ""),
                row.get("doc_text", ""),
                profile,
            ))

        return self.execute_many(sql, params)

    def query_mainindex(
        self,
        doc_ids: List[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """查询 MainIndex"""
        if not self.ensure_connection():
            return []

        rdb_cfg = self.cfg.rdb
        self.use_database(rdb_cfg.database)

        if doc_ids:
            placeholders = ", ".join(["%s"] * len(doc_ids))
            sql = f"SELECT * FROM `{rdb_cfg.table}` WHERE doc_id IN ({placeholders}) LIMIT {limit} OFFSET {offset}"
            return self.execute(sql, tuple(doc_ids))
        else:
            sql = f"SELECT * FROM `{rdb_cfg.table}` LIMIT {limit} OFFSET {offset}"
            return self.execute(sql)

    def get_dimension_columns(self) -> List[str]:
        """获取所有 dim_* 列名"""
        if not self.ensure_connection():
            return []

        rdb_cfg = self.cfg.rdb
        self.use_database(rdb_cfg.database)
        rows = self.execute(f"SHOW COLUMNS FROM `{rdb_cfg.table}`")
        return [r["Field"] for r in rows if r["Field"].startswith("dim_")]

    def add_dimension_column(self, dim_name: str, table: str = None) -> bool:
        """新增维度列"""
        if not self.ensure_connection():
            return False

        rdb_cfg = self.cfg.rdb
        tbl = table or rdb_cfg.table
        self.use_database(rdb_cfg.database)

        col_name = f"dim_{dim_name}"
        if self.table_exists(tbl):
            cols = [r["Field"] for r in self.describe_table(tbl)]
            if col_name not in cols:
                try:
                    self.execute(f"ALTER TABLE `{tbl}` ADD COLUMN `{col_name}` TEXT")
                    self.commit()
                    print(f"[MySQL] 新增维度列: {col_name}")
                    return True
                except Exception as e:
                    print(f"[MySQL] 新增维度列失败: {e}")
        return False

    # ==================== 提示优化表（来自 code1）====================

    def init_prompt_tables(self, force: bool = False) -> Dict[str, Any]:
        """初始化提示优化相关表"""
        if not self.ensure_connection():
            return {"status": "error", "message": "MySQL 未连接"}

        rdb_cfg = self.cfg.rdb
        self.create_database_if_not_exists(rdb_cfg.database)
        self.use_database(rdb_cfg.database)

        tables = {}

        # prompt_clusters: 聚类优化后的提示
        if self.table_exists("prompt_clusters") and force:
            self.execute("DROP TABLE prompt_clusters")
        if not self.table_exists("prompt_clusters"):
            self.execute("""
                CREATE TABLE prompt_clusters (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    cluster_id INT,
                    cluster_name VARCHAR(255),
                    question_count INT DEFAULT 0,
                    optimized_prompt TEXT,
                    optimization_reason TEXT,
                    source_questions TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            tables["prompt_clusters"] = "created"
        else:
            tables["prompt_clusters"] = "exists"

        # prompt_iterations: 迭代优化历史
        if self.table_exists("prompt_iterations") and force:
            self.execute("DROP TABLE prompt_iterations")
        if not self.table_exists("prompt_iterations"):
            self.execute("""
                CREATE TABLE prompt_iterations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    question_id VARCHAR(255),
                    iteration_round INT,
                    generated_answer TEXT,
                    reference_answer TEXT,
                    improvement_suggestion TEXT,
                    optimized_prompt TEXT,
                    prompt_template TEXT,
                    key_aspects TEXT,
                    scene_analysis TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            tables["prompt_iterations"] = "created"
        else:
            tables["prompt_iterations"] = "exists"

        # tourist_questions: 旅游问答数据集
        if self.table_exists("tourist_questions") and force:
            self.execute("DROP TABLE tourist_questions")
        if not self.table_exists("tourist_questions"):
            self.execute("""
                CREATE TABLE tourist_questions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    question_id VARCHAR(255) UNIQUE,
                    attraction VARCHAR(255),
                    question TEXT,
                    answer TEXT,
                    document TEXT,
                    doc_id VARCHAR(255),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            tables["tourist_questions"] = "created"
        else:
            tables["tourist_questions"] = "exists"

        self.commit()
        return {"status": "success", "tables": tables}

    def insert_prompt_clusters(self, rows: List[Dict]) -> int:
        """写入聚类提示"""
        if not rows:
            return 0
        self.use_database(self.cfg.rdb.database)

        sql = """
        INSERT INTO prompt_clusters
            (cluster_id, cluster_name, question_count, optimized_prompt,
             optimization_reason, source_questions)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            question_count = VALUES(question_count),
            optimized_prompt = VALUES(optimized_prompt),
            optimization_reason = VALUES(optimization_reason),
            source_questions = VALUES(source_questions)
        """
        params = [
            (
                r.get("cluster_id", 0),
                r.get("cluster_name", ""),
                r.get("question_count", 0),
                r.get("optimized_prompt", ""),
                r.get("optimization_reason", ""),
                json.dumps(r.get("source_questions", []), ensure_ascii=False),
            )
            for r in rows
        ]
        return self.execute_many(sql, params)

    def insert_prompt_iterations(self, rows: List[Dict]) -> int:
        """写入迭代优化历史"""
        if not rows:
            return 0
        self.use_database(self.cfg.rdb.database)

        sql = """
        INSERT INTO prompt_iterations
            (question_id, iteration_round, generated_answer, reference_answer,
             improvement_suggestion, optimized_prompt, prompt_template,
             key_aspects, scene_analysis)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            generated_answer = VALUES(generated_answer),
            improvement_suggestion = VALUES(improvement_suggestion),
            optimized_prompt = VALUES(optimized_prompt)
        """
        params = [
            (
                r.get("question_id", ""),
                r.get("iteration_round", 0),
                r.get("generated_answer", ""),
                r.get("reference_answer", ""),
                r.get("improvement_suggestion", ""),
                r.get("optimized_prompt", ""),
                r.get("prompt_template", ""),
                json.dumps(r.get("key_aspects", []), ensure_ascii=False),
                r.get("scene_analysis", ""),
            )
            for r in rows
        ]
        return self.execute_many(sql, params)

    def insert_tourist_questions(self, rows: List[Dict]) -> int:
        """写入旅游问答数据集"""
        if not rows:
            return 0
        self.use_database(self.cfg.rdb.database)

        sql = """
        INSERT INTO tourist_questions
            (question_id, attraction, question, answer, document, doc_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            attraction = VALUES(attraction),
            question = VALUES(question),
            answer = VALUES(answer),
            document = VALUES(document)
        """
        params = [
            (
                r.get("question_id", ""),
                r.get("attraction", ""),
                r.get("question", ""),
                r.get("answer", ""),
                r.get("document", ""),
                r.get("doc_id", ""),
            )
            for r in rows
        ]
        return self.execute_many(sql, params)

    def query_prompt_clusters(self) -> List[Dict]:
        """查询所有聚类提示"""
        if not self.ensure_connection():
            return []
        self.use_database(self.cfg.rdb.database)
        return self.execute("SELECT * FROM prompt_clusters ORDER BY cluster_id")

    def query_prompt_iterations(self, question_id: str = None) -> List[Dict]:
        """查询迭代历史"""
        if not self.ensure_connection():
            return []
        self.use_database(self.cfg.rdb.database)
        if question_id:
            return self.execute(
                "SELECT * FROM prompt_iterations WHERE question_id = %s ORDER BY iteration_round",
                (question_id,)
            )
        return self.execute("SELECT * FROM prompt_iterations LIMIT 100")

    def query_tourist_questions(self, limit: int = 50) -> List[Dict]:
        """查询旅游问答数据"""
        if not self.ensure_connection():
            return []
        self.use_database(self.cfg.rdb.database)
        return self.execute(f"SELECT * FROM tourist_questions LIMIT {limit}")


# ==================== 单例管理器 ====================

_mysql_instance: Optional[MySQLService] = None


def get_mysql() -> MySQLService:
    global _mysql_instance
    if _mysql_instance is None:
        _mysql_instance = MySQLService()
    return _mysql_instance


def init_mysql(config=None) -> MySQLService:
    global _mysql_instance
    _mysql_instance = MySQLService(config)
    return _mysql_instance
