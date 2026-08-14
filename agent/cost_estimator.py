import psycopg
from multiprocessing import Queue
import json

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
        import time
        start_total = time.time()
        costs = [0 for _ in range(self.n_templates)]
        try:
            conn = psycopg.connect(self.connection_string)
        except Exception as e:
            print(f"CostEstimator: connection error: {e}")
            self.queue.put(costs)
            return

        conn_time = time.time() - start_total
        print(f"[TIMER CostEstimator] connection took {conn_time:.2f}s")

        with conn.cursor() as cur:
            indexes_required = 0

            if indexes is not None:
                for index in indexes:
                    table = index[0]
                    columns = index[1]
                    indexes_required += 1
                    creation_string = 'CREATE INDEX candidate_index_%d ON %s (%s)' % (indexes_required, table, ', '.join(columns))
                    try:
                        cur.execute('SELECT indexrelid FROM hypopg_create_index($$%s$$);' % creation_string)
                    except Exception as e:
                        print(f"CostEstimator: virtual index creation error: {e}")
                        conn.rollback()

            for idx, query in enumerate(queries):
                q_start = time.time()

                for statement in query.split(';'):
                    statement = statement.lower()
                    if 'create view' in statement or 'drop view' in statement:
                        try:
                            cur.execute(statement)
                        except Exception as e:
                            print(f"CostEstimator: DDL error: {e}")
                            conn.rollback()
                    elif 'select' in statement or 'update' in statement or 'insert' in statement or 'delete' in statement:
                        try:
                            cur.execute('EXPLAIN (FORMAT JSON) %s' % statement)
                            row = cur.fetchone()
                            if row and row[0]:
                                try:
                                    # parse JSON
                                    data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                                    if isinstance(data, list) and len(data) > 0:
                                        plan = data[0].get('Plan', {})
                                        total_cost = plan.get('Total Cost')
                                        if total_cost is not None:
                                            costs[templates[idx]] += float(total_cost)
                                except Exception as e:
                                    print(f"CostEstimator: JSON parsing error: {e}")
                        except Exception as e:
                            print(f"CostEstimator: EXPLAIN error: {e}")
                            conn.rollback()
                print(f"[TIMER CostEstimator] query {idx} took {time.time() - q_start:.2f}s")

            conn.commit()

        total = time.time() - start_total
        print(f"[TIMER CostEstimator] total run time: {total:.2f}s for {len(queries)} queries")

        self.queue.put(costs)