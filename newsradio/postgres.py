"""Small compatibility layer for the fixed SQL statements in db.py, not a general SQL translator."""
import re


class Row(dict):
    def __getitem__(self, key):
        return list(self.values())[key] if isinstance(key, int) else super().__getitem__(key)


def row_factory(cursor):
    names = [col.name for col in cursor.description] if cursor.description else []
    return lambda values: Row(zip(names, values))


class Cursor:
    def __init__(self, cursor, inserted=False):
        self.cursor = cursor
        self.rowcount = cursor.rowcount
        row = cursor.fetchone() if inserted else None
        self.lastrowid = row[0] if row else None

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()


class PostgresConnection:
    def __init__(self, url, schema):
        import psycopg
        self.raw = psycopg.connect(url, row_factory=row_factory)
        schema = schema.replace('INTEGER PRIMARY KEY', 'BIGSERIAL PRIMARY KEY')
        schema = re.sub(r'\bREAL\b', 'DOUBLE PRECISION', schema)
        schema = re.sub(r'\bBLOB\b', 'BYTEA', schema)
        for statement in schema.split(';'):
            if statement.strip():
                self.raw.execute(statement)
        self.raw.commit()

    def execute(self, sql, params=()):
        sql = sql.replace('?', '%s')
        inserted = sql.lstrip().startswith('INSERT')
        if 'INSERT OR IGNORE' in sql:
            sql = sql.replace('INSERT OR IGNORE', 'INSERT') + ' ON CONFLICT DO NOTHING'
        if inserted:
            sql += ' RETURNING id'
        try:
            return Cursor(self.raw.execute(sql, params), inserted)
        except Exception:
            self.raw.rollback()
            raise

    def commit(self):
        self.raw.commit()

    def close(self):
        self.raw.close()
