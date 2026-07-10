import numpy as np
import psycopg
from agent.database import Replica

class CandidateLimitingHeuristic:
    def __init__(self, replica: Replica, candidates: list[tuple[str, str]], workload: list[str], n_candidates: int):
        '''
        If we only want to send the best *n* candidates to the index recommendation algorithm,
        how do we do that? With a Candidate Limiting Heuristic, of course !

        Invokes the cost estimator to estimate the relative performance of each single-column index,
        then returns the best *n* options. If `n_candidates` is greater than the number of candidates
        passed to the constructor, simply returns all of the candidates.
        
        :param replica: a database replica object to invoke the cost estimator
        :param candidates: the columns/tables to construct the index on
        :param workload: the queries to cost-estimate to run the heuristic
        :param n_candidates: how many candidates should the heuristic return?
        '''
        self.replica = replica
        self.candidates = candidates
        self.workload = workload
        self.n_candidates = n_candidates
    
    def get_candidates(self):
        if self.n_candidates >= len(self.candidates):
            return self.candidates
        
        costs = []
        conn = psycopg.connect(self.replica.connection_string())

        for column, table in self.candidates:
            costs.append(self._evaluate_one(column, table, conn))

        conn.close()
        best = np.argsort(costs)
        top_n = best[:self.n_candidates]
        selected = []

        for i in top_n:
            selected.append(self.candidates[i][0])
        
        return selected

    def _evaluate_one(self, column: str, table: str, conn: psycopg.Connection):
        '''
        Evaluate a single index candidate's predicted cost.

        :param column: the column to create the index on
        :param table: which table that column belongs to
        :param cost: the connection to the database to be used
        :returns cost: the predicted cost for this column's index alone
        '''
        cost = 0
        with conn.cursor() as cur:
            cur.execute(f'SELECT indexrelid FROM hypopg_create_index($$CREATE INDEX test_idx ON {table} ({', '.join(column)})$$)')
            for query in self.workload:
                for statement in query.split(';'):
                    if 'create view' in statement or 'drop view' in statement:
                        cur.execute(statement)
                    elif 'select' in statement:
                        cur.execute('EXPLAIN (FORMAT JSON) %s' % statement)
                        if after_timing := cur.fetchone()[0][0]['Plan']['Total Cost']:
                            cost += float(after_timing)
            cur.execute('SELECT hypopg_reset()')
        
        return cost
