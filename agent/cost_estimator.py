import psycopg
import re
import json
from multiprocessing import Queue

class CostEstimator:
    def __init__(self, n_templates: int, connection_string: str, queue: Queue):
        self.n_templates = n_templates
        self.connection_string = connection_string
        self.queue = queue

    def run(self, queries: list[str], templates: list, indexes: list):
        costs = [0.0 for _ in range(self.n_templates)]
        
        try:
            conn = psycopg.connect(self.connection_string)
        except Exception as conn_err:
            print(f"[CostEstimator Error] Cannot connect to the database: {conn_err}")
            self.queue.put(costs)
            return

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
                        print(f"[CostEstimator] Virtual index creation error: {e}")
                        conn.rollback()

            for idx, query in enumerate(queries):
                if not query:
                    continue
                
                query_str = str(query)
                
                query_clean = re.sub(r'--.*$', '', query_str, flags=re.MULTILINE)
                
                query_clean = re.sub(r'\s*\(\d+\)\s*', ' ', query_clean)
                
                query_clean = re.sub(r'(?i)\bset\s+rowcount\s+[-\d]+\b', '', query_clean)
                query_clean = re.sub(r'(?i)\bgo\b', '', query_clean)
                
                query_clean = query_clean.strip()
                
                if query_clean.endswith(';'):
                    query_clean = query_clean[:-1].strip()
                    
                if not query_clean:
                    continue

                statement_lower = query_clean.lower()
                
                if 'create view' in statement_lower or 'drop view' in statement_lower:
                    try:
                        cur.execute(query_clean)
                    except psycopg.errors.Error:
                        conn.rollback()
                        continue
                elif any(cmd in statement_lower for cmd in ['select', 'update', 'insert', 'delete']):
                    try:
                        cur.execute('EXPLAIN (FORMAT JSON) %s;' % query_clean)
                        row = cur.fetchone()
                        
                        if row and row[0]:
                            raw_json = row[0]
                            
                            parsed_data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
                            
                            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                                plan_dict = parsed_data[0].get('Plan', {})
                                if 'Total Cost' in plan_dict:
                                    total_cost_val = float(plan_dict['Total Cost'])
                                    
                                    try:
                                        t_idx = int(templates[idx])
                                        if 0 <= t_idx < len(costs):
                                            costs[t_idx] += total_cost_val
                                        else:
                                            costs[0] += total_cost_val
                                    except Exception:
                                        costs[0] += total_cost_val
                                        
                    except psycopg.errors.Error:
                        conn.rollback()
                        continue
                    except Exception:
                        continue

            conn.commit()
            conn.close()
        
        if sum(costs) == 0.0:
            print(f"[CostEstimator Debug Warning] The total cost for this batch is 0.0 ! Number of queries evaluated: {len(queries)}")
            
        self.queue.put(costs)