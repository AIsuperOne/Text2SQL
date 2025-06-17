import pymysql
import pandas as pd
from config.settings import MYSQL_CONFIG

class MySQLConnector:
    def __init__(self):
        self.conn = pymysql.connect(
            host=MYSQL_CONFIG["host"],
            port=MYSQL_CONFIG["port"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            database=MYSQL_CONFIG["database"],
            charset="utf8mb4"
        )

    def execute_query(self, sql):
        df = pd.read_sql(sql, self.conn)
        return df
    
    def get_schema_info(self):
        """获取数据库的完整结构信息"""
        # 获取所有表
        tables_query = """
        SELECT TABLE_NAME, TABLE_COMMENT 
        FROM information_schema.TABLES 
        WHERE TABLE_SCHEMA = %s
        """
        tables_df = pd.read_sql(tables_query, self.conn, params=[MYSQL_CONFIG["database"]])
        
        schema_info = []
        for _, table in tables_df.iterrows():
            table_name = table['TABLE_NAME']
            table_comment = table['TABLE_COMMENT'] or ''
            
            # 获取表的列信息
            columns_query = """
            SELECT COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT 
            FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
            """
            columns_df = pd.read_sql(columns_query, self.conn, 
                                     params=[MYSQL_CONFIG["database"], table_name])
            
            # 构建DDL语句
            ddl = f"CREATE TABLE {table_name} ("
            column_defs = []
            for _, col in columns_df.iterrows():
                col_def = f"{col['COLUMN_NAME']} {col['DATA_TYPE']}"
                if col['COLUMN_COMMENT']:
                    col_def += f" COMMENT '{col['COLUMN_COMMENT']}'"
                column_defs.append(col_def)
            
            ddl += ", ".join(column_defs) + ")"
            if table_comment:
                ddl += f" COMMENT='{table_comment}'"
            
            schema_info.append({
                "table_name": table_name,
                "table_comment": table_comment,
                "ddl": ddl,
                "columns": columns_df.to_dict('records')
            })
        
        return schema_info
    
    def get_sample_data(self, table_name, limit=5):
        """获取表的样例数据"""
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return pd.read_sql(query, self.conn)
        except:
            return pd.DataFrame()


