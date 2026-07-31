import psycopg

class Replica:
    def __init__(self, id, hostname, port, dbname, user, password):
        """Initialize a database replica object.

        Args:
            id (int): Unique identifier for this replica.
            hostname (str): Database server hostname.
            port (int): Database server port.
            dbname (str): Name of the database.
            user (str): Connection user.
            password (str): Connection password.
        """
        self.id = id
        self.hostname = hostname
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.conn = None

    def connection_string(self):
        """Build a PostgreSQL connection string from the replica's attributes.

        Returns:
            str: The connection string in the format 'host=... port=... dbname=... user=... password=...'.
        """
        return f'host={self.hostname} port={self.port} dbname={self.dbname} user={self.user} password={self.password}'
    
    def drop_all_indexes(self, tables, mode: str):
        """Drop all indexes (real or hypothetical) on the given tables.

        If mode is 'cost', resets HypoPG hypothetical indexes. Otherwise, drops
        real indexes using a PL/pgSQL script.

        Args:
            tables (list[str]): List of table names.
            mode (str): Either 'cost' (use HypoPG reset) or other (drop real indexes).
        """
        if not tables:
            return
        try:
            conn = self.connection()
            with conn.cursor() as cur:
                for table in tables:
                    if mode == 'cost':
                        cur.execute('SELECT hypopg_reset();')
                    else:
                        # https://stackoverflow.com/questions/34010401/how-can-i-drop-all-indexes-of-a-table-in-postgres
                        cur.execute(QUERY_TEMPLATE % table)
        except Exception as e:
            print(f'error while trying to drop indexes in replica {self.id}!')
            print(e)


    def connection(self):
        """Establish or return the existing database connection.

        Returns:
            psycopg.Connection: The active connection object.
        """
        if self.conn is None:
            self.conn = psycopg.connect(self.connection_string())
        return self.conn
    
    def commit(self):
        """Commit the current transaction on the database connection."""
        self.conn.commit()

    def rollback(self):
        """Roll back the current transaction on the database connection."""
        self.conn.rollback()
    
    def close(self):
        """Close the database connection and set the connection attribute to None."""
        self.conn.close()
        self.conn = None


QUERY_TEMPLATE = '''
DO
$do$
DECLARE
   _sql text;
BEGIN   
   SELECT 'DROP INDEX ' || string_agg(indexrelid::regclass::text, ', ')
   FROM   pg_index  i
   LEFT   JOIN pg_depend d ON d.objid = i.indexrelid
                          AND d.deptype = 'i'
   WHERE  i.indrelid = '%s'::regclass  -- possibly schema-qualified
   AND    d.objid IS NULL                      -- no internal dependency
   INTO   _sql;
   
   IF _sql IS NOT NULL THEN                    -- only if index(es) found
     EXECUTE _sql;
   END IF;
END
$do$;
'''