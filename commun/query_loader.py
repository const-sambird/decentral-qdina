import glob
import os
import random

def load_training_set_queries(path: str, fraction: float) -> tuple[list[str], list[int]]:
    '''
    Load `fraction` of the queries in the training set located at `path`.
    Returns the query text in a list as well as which template each query belongs to.
    
    :param path: the location of the training set
    :param fraction: what proportion of the total training set should we load
    :returns queries: the query text
    :returns templates: which template each query belongs to
    '''
    all_queries = glob.glob(f'{path}/*.sql')
    n_queries = len(all_queries)
    n_selected = round(n_queries * fraction)
    query_names = [os.path.basename(q) for q in all_queries]
    selections = random.sample(query_names, n_selected)
    selections = sorted(selections)

    queries = []
    templates = []

    for selection in selections:
        template = selection.split('_')[0]
        template = int(template)
        with open(f'{path}/{selection}', 'r') as infile:
            lines = infile.readlines()
            if lines[0].startswith('--'):
                lines = lines[1:]
            flattened = ' '.join(lines)
            queries.append(flattened.replace('\n', ' ').replace('\t', ' '))
            templates.append(template - 1)

    return queries, templates

def load_low_data_queries(path: str, queries_per_template: str) -> tuple[list[str], list[int]]:
    '''
    Loads `queries_per_template` instances of each query template in the training set found
    at `path`. The low data scenario guarantees a workload uniformly distributed amongst all
    of the query templates, so we only need to control the number of queries from each template
    that we load.

    :param path: the location of the training set
    :param queries_per_template: how many queries we should be drawing from each template
    :returns queries: the query text
    :returns templates: which template each query belongs to
    '''
    # first: how many templates are there, and how many queries do each one of them have?
    all_queries = glob.glob(f'{path}/*.sql')
    query_names = [os.path.basename(q) for q in all_queries]
    name_parts = [q.split('_') for q in query_names]
    template_strs = list(set([t[0] for t in name_parts]))
    query_nums = list(set([int(q[1].strip('.sql')) for q in name_parts]))

    # next: actually sample
    queries = []
    templates = []
    nums_to_load = [[] for _ in range(len(template_strs))]

    for i in range(len(template_strs)):
        try:
            nums_to_load[i] = random.sample(query_nums, queries_per_template)
        except ValueError:
            raise ValueError(f'couldn\'t select the requested {queries_per_template} queries from each template; there are only {len(query_nums)} to pick from')
    
    for i, template in enumerate(template_strs):
        for query_num in nums_to_load[i]:
            with open(f'{path}/{template}_{query_num}.sql', 'r') as infile:
                lines = infile.readlines()
            flattened = ' '.join(lines[1:])
            queries.append(flattened.replace('\n', ' ').replace('\t', ' '))
            templates.append(int(template) - 1)
    
    return queries, templates

def load_candidates(path: str) -> list[tuple[str]]:
    '''
    If the query candidates are given explicity in a file on disk
    (as opposed to parsed from the workload), this function reads
    in those candidates and returns them to the preprocessor.
    '''
    candidates = []

    with open(path, 'r') as file:
        candidates = file.readline()
        candidates = candidates.split(' ')
        candidates = [tuple(c.split(',')) for c in candidates]

    return candidates
