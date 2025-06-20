import pandas as pd
from sqlalchemy import create_engine
from config.settings import MYSQL_CONFIG

class MySQLConnector:
    def __init__(self):
        # 构建 SQLAlchemy 连接字符串
        # 格式: 'mysql+pymysql://user:password@host:port/database'
        db_uri = (
            f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
            f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
        )
        
        # 创建 SQLAlchemy 引擎
        self.engine = create_engine(db_uri)
        
        # 不再需要 self.conn
        # self.conn = pymysql.connect(...)

    def execute_query(self, sql, params=None):
        # 使用 SQLAlchemy 引擎执行查询，这是 Pandas 推荐的方式
        # 警告会消失
        df = pd.read_sql(sql, self.engine, params=params)
        return df
    
    def get_schema_info(self):
        """获取数据库的完整结构信息"""
        # 获取所有表
        tables_query = """
        SELECT TABLE_NAME, TABLE_COMMENT 
        FROM information_schema.TABLES 
        WHERE TABLE_SCHEMA = %(db_name)s
        """
        tables_df = self.execute_query(tables_query, params={"db_name": MYSQL_CONFIG["database"]})
        
        schema_info = []
        for _, table in tables_df.iterrows():
            table_name = table['TABLE_NAME']
            table_comment = table['TABLE_COMMENT'] or ''
            
            # 获取表的列信息
            columns_query = """
            SELECT COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT 
            FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = %(db_name)s AND TABLE_NAME = %(tbl_name)s
            ORDER BY ORDINAL_POSITION
            """
            columns_df = self.execute_query(
                columns_query, 
                params={"db_name": MYSQL_CONFIG["database"], "tbl_name": table_name}
            )
            
            # 构建DDL语句
            ddl = f"CREATE TABLE `{table_name}` ("
            column_defs = []
            for _, col in columns_df.iterrows():
                col_def = f"`{col['COLUMN_NAME']}` {col['DATA_TYPE']}"
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
            # 使用 text() 来防止SQL注入，并正确格式化表名
            from sqlalchemy import text
            query = f"SELECT * FROM `{table_name}` LIMIT :limit"
            return self.execute_query(query, params={"limit": limit})
        except Exception as e:
            print(f"获取样例数据失败 for table {table_name}: {e}")
            return pd.DataFrame()



