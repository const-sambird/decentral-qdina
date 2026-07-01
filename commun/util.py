from itertools import chain, combinations
from random import randrange
import re
import os

def powerset(iterable, max_len):
    # https://docs.python.org/2/library/itertools.html#recipes
    s = list(iterable)
    if max_len is not None:
        return chain.from_iterable(combinations(s, r) for r in range(min(len(s), max_len)+1))
    else:
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def extract_table_from_query(query: str) -> str | None:
        REGEX = '^( )*(([a-zA-Z0-9]|\\.|_|-)+)( )*'
        split = re.split('FROM ', query, flags=re.IGNORECASE)
        
        #if len(split) != 2:
        #    return None # subquery or malformed
        if table_name := re.search(REGEX, split[1], re.IGNORECASE):
            return table_name.group(1)
        else:
            return None
        
def extract_table_from_query_aa(query, coldict):
    for t, cols in coldict.items():
        if (query,) in cols:
            return t
        
def extract_columns_from_query(query, coldict):
    REGEX = 'WHERE (.+?)(?:\\)|group by|order by|;)'
    columns = {}

    predicates = re.findall(REGEX, query, re.IGNORECASE)

    for predicate in predicates:
        for column, table in coldict.items():
            if column in predicate:
                if table not in columns:
                    columns[table] = set()
                columns[table].add(column)
    
    return columns

def construct_indexes_from_candidate(candidate: tuple[str], cols_to_table: dict[str, str]) -> dict[str, list[str]]:
    '''
    Given a list of columns to index over, returns a dict
    of those columns sorted into the tables they belong to.
    '''
    candidate_representation = {}
    for column in candidate:
        table = cols_to_table[column]
        if table not in candidate_representation:
            candidate_representation[table] = []
        candidate_representation[table].append(column)
    return candidate_representation

def insert_dummy_values(template):
    HASH_VALUES = '\\$?\\@\\@\\@'
    STRING_VALUES = '\\$?\\&\\&\\&'
    INT_VALUES = '\\$?\\^\\^\\^'

    template = re.sub(HASH_VALUES, "'hash'", template)
    template = re.sub(STRING_VALUES, "'foo'", template)
    template = re.sub(INT_VALUES, '-1', template)

    return template

def drop_one(from_tuple):
    els = list(from_tuple)
    to_drop = randrange(0, len(els))
    del els[to_drop]
    return tuple(els)

def update_query_text(text: str) -> str:
    '''
    Updates query text to work in PostgreSQL.

    Taken from https://github.com/hyrise/index_selection_evaluation

    :param text: the text of the query to update
    :returns text: the corrected version
    '''
    text = text.replace(";\nlimit ", " limit ").replace("limit -1", "")
    text = re.sub(r" ([0-9]+) days\)", r" interval '\1 days')", text)
    text = add_alias_subquery(text)
    return text

# PostgreSQL requires an alias for subqueries
def add_alias_subquery(query_text):
    text = query_text.lower()
    positions = []
    for match in re.finditer(r"((from)|,)[  \n]*\(", text):
        counter = 1
        pos = match.span()[1]
        while counter > 0:
            char = text[pos]
            if char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
            pos += 1
        next_word = query_text[pos:].lstrip().split(" ")[0].split("\n")[0]
        if next_word[0] in [")", ","] or next_word in [
            "limit",
            "group",
            "order",
            "where",
        ]:
            positions.append(pos)
    for pos in sorted(positions, reverse=True):
        query_text = query_text[:pos] + " as alias123 " + query_text[pos:]
    return query_text


def parse_replicas_csv(file_path: str) -> dict:
    '''
    Parses the replicas.csv file to load cluster topology configuration.
    Each valid line format: id,hostname,port,dbname,user,password
    '''
    replicas_config = {}
    if not os.path.exists(file_path):
        return replicas_config
        
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip headers or empty lines
            if not line or line.startswith('id'):
                continue
            parts = line.split(',')
            if len(parts) >= 6:
                r_id = int(parts[0])
                replicas_config[r_id] = {
                    'hostname': parts[1],
                    'port': int(parts[2]),
                    'dbname': parts[3],
                    'user': parts[4],
                    'password': parts[5]
                }
    return replicas_config