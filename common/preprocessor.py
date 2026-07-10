import random
from common.util import extract_columns_from_query, construct_indexes_from_candidate, drop_one, powerset
from common.query_loader import load_candidates
from itertools import permutations
from common.profiling import Profiler
from agent.database import Replica
from common.heuristic import CandidateLimitingHeuristic

class Preprocessor:
    def __init__(self, profiler: Profiler, database: Replica, max_index_width: int, queries: list[str], templates: list[int]):
        '''
        Instantiate the preprocessing module.

        `database` is the database replica we should use for querying table names,
        column names, and index sizes.
        '''
        self.tables = []
        self.columns = []
        self.workload = queries
        self.template_assignments = templates
        self.profiler = profiler
        self.database = database
        self.max_index_width = max_index_width

    def preprocess(self, candidate_path: str | None, max_candidates: int | None):
        self.profiler.time_in('filesystem')
        self.templates = []
        for x in set(self.template_assignments):
            self.templates.append(self.workload[self.template_assignments.index(x)])
        self.profiler.time_out()
        self.profiler.time_in('database.preprocess')
        self._read_tables()
        self._read_columns()
        self.profiler.time_out()
        if candidate_path is None:
            self.get_indexable_columns(self.templates)
        else:
            self.candidates = load_candidates(candidate_path)
        
        if max_candidates is not None:
            self.limit_candidate_size(max_candidates)

        print(self.candidates)

    def _read_tables(self):
        conn = self.database.connection()
        with conn.cursor() as cur:
            cur.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\';')
            for name in cur.fetchall():
                if 'hypopg' not in name[0]:
                    self.tables.append(name[0])
            conn.commit()
    
    def _read_columns(self):
        assert len(self.tables) > 0, 'trying to read columns before tables!'

        self.columns = []
        self.cols_to_table = {}

        QUERY_TEMPLATE = "SELECT * FROM %s LIMIT 0;"

        try:
            conn = self.database.connection()
            with conn.cursor() as cur:
                for table in self.tables:
                    cur.execute(QUERY_TEMPLATE % table)
                    for desc in cur.description:
                        self.columns.append(desc[0])
                        self.cols_to_table[desc[0]] = table
                conn.commit()
        except Exception as err:
            print('got an exception in the database connection')
            print(err)
            conn.rollback()
    
    def get_indexable_columns(self, templates):
        self.candidates = {}

        for idx, template in enumerate(templates):
            matches = extract_columns_from_query(template, self.cols_to_table)
            for table, columns in matches.items():
                if table not in self.candidates:
                    self.candidates[table] = set()
                for index in powerset(sorted(columns), self.max_index_width):
                    if len(index) == 0: continue
                    for permutation in permutations(index):
                        self.candidates[table].add(permutation)
        
        # flatten dict of sets of tuples into a list of tuples
        self.tables = list(self.candidates.keys())
        self.candidates = list(set([x for v in self.candidates.values() for x in v]))
        self.candidates = sorted(self.candidates)
    
    def limit_candidate_size(self, max_candidates):
        if max_candidates >= len(self.candidates): return
        
        candidates = list(zip(self.candidates, [self.cols_to_table[c[0]] for c in self.candidates]))
        heuristic = CandidateLimitingHeuristic(self.database, candidates, self.workload, max_candidates)
        self.candidates = heuristic.get_candidates()
