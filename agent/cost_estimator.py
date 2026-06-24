import psycopg
from multiprocessing import Queue

class CostEstimator:
    def __init__(self, n_templates: int, connection_string: str, queue: Queue):
        '''
        Invokes the PostgreSQL cost estimation module on a set of queries on a given
        replica. Puts the estimated costs back in a queue, as this is intended to be
        run in a multiprocessing context.

        :param n_templates: the number of unique templates for which to estimate costs
        :param connection_string: the database replica string to open a connection to
        :param queue: the queue to return estimated costs into
        '''
        self.n_templates = n_templates
        self.connection_string = connection_string
        self.queue = queue

    def run(self, queries: list[str], templates: list[int], indexes: list):
        '''
        Estimate costs for the queries, and put those costs into the
        Queue passed to the constructor.

        :param queries: the queries in the workload to estimate
        :param templates: which template each query belongs to
        :param indexes: a description of the indexes to simulate for cost estimation
        '''
        costs = [0 for _ in range(self.n_templates)]
        conn = psycopg.connect(self.connection_string)
        with conn.cursor() as cur:
            indexes_required = 0

            if indexes is not None:
                for index in indexes:
                    table = index[0]
                    columns = index[1]
                    indexes_required += 1
                    creation_string = 'CREATE INDEX candidate_index_%d ON %s (%s)' % (indexes_required, table, ', '.join(columns))
                    cur.execute('SELECT indexrelid FROM hypopg_create_index($$%s$$);' % creation_string)

            for idx, query in enumerate(queries):
                for statement in query.split(';'):
                    statement = statement.lower()
                    if 'create view' in statement or 'drop view' in statement:
                        cur.execute(statement)
                    elif 'select' in statement or 'update' in statement or 'insert' in statement or 'delete' in statement:
                        cur.execute('EXPLAIN (FORMAT JSON) %s' % statement)
                        if after_timing := cur.fetchone()[0][0]['Plan']['Total Cost']:
                            costs[templates[idx]] += float(after_timing)

            conn.commit()
        
        self.queue.put(costs)