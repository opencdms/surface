import pandas as pd
from tempestas_api import settings
import psycopg2

df = pd.read_csv('3-02.csv')
data = df.to_dict('records')
with psycopg2.connect(settings.SURFACE_CONNECTION_STRING) as conn:
    with conn.cursor() as cursor:
        query = """
            INSERT INTO wx_country(created_at, updated_at, name, notation, description)
            VALUES (CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %(name)s, %(notation)s, %(description)s)
            ON CONFLICT (name) DO UPDATE
            SET updated_at = CURRENT_TIMESTAMP,
                notation = %(notation)s,
                description = %(description)s
            """
        cursor.executemany(query, data)
    conn.commit()