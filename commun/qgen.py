import subprocess
import os
import random

BENCH_TYPE = 'tpc-h'
# BENCH_TYPE = 'tpc-ds'
OUTPUT_TO = 'train'
# OUTPUT_TO = 'test'
NUM_INSTANCES = 10

if BENCH_TYPE == 'tpc-h':
    ROOTDIR = f'/proj/qdina-PG0/dina-set/h/{OUTPUT_TO}'
    DBGEN_DIR = '/users/sambird/tpc-h/dbgen'
    NUM_TEMPLATES = 22
    SCALE_FACTOR = '10'
elif BENCH_TYPE == 'tpc-ds':
    ROOTDIR = f'/proj/qdina-PG0/dina-set/ds/{OUTPUT_TO}'
    DBGEN_DIR = '/users/sambird/tpc-ds/tools'
    NUM_TEMPLATES = 99
    SCALE_FACTOR = '10'
SEED_BASE = random.randint(1_000_000_000, 9_999_999_999)

def _compile_qgen():
    subprocess.run('make', cwd=DBGEN_DIR)

def _create_tpch_queries():
    for i in range(NUM_TEMPLATES):
        for j in range(NUM_INSTANCES):
            with open(f'{ROOTDIR}/{i + 1}_{j}.sql', 'w') as outfile:
                subprocess.run([f'{DBGEN_DIR}/qgen', '-s', SCALE_FACTOR, '-r', str(SEED_BASE + j), str(i + 1)],
                               cwd=DBGEN_DIR,
                               env=dict(os.environ, DSS_QUERY=f'{DBGEN_DIR}/queries'),
                               stdout=outfile)

def _compile_dsqgen():
    subprocess.run(['make', 'CC="gcc-9"'], cwd=DBGEN_DIR)

def _create_tpcds_queries():
    for i in range(NUM_TEMPLATES):
        for j in range(NUM_INSTANCES):
            with open(f'{ROOTDIR}/{i + 1}_{j}.sql', 'w') as outfile:
                subprocess.run([f'{DBGEN_DIR}/dsqgen',
                                '-SCALE', SCALE_FACTOR,
                                '-RNGSEED', str(SEED_BASE + j),
                                '-TEMPLATE', f'query{i + 1}.tpl',
                                '-DIALECT', 'netezza',
                                '-DIRECTORY', os.path.normpath(os.path.join(DBGEN_DIR, '..', 'query_templates')),
                                '-FILTER', 'Y'],
                            cwd=DBGEN_DIR,
                            stdout=outfile)

def tpch():
    _compile_qgen()
    print(f'TPC-H: writing {NUM_INSTANCES} queries per template at sf {SCALE_FACTOR} to {ROOTDIR}')
    _create_tpch_queries()

def tpcds():
    _compile_dsqgen()
    print(f'TPC-DS: writing {NUM_INSTANCES} queries per template at sf {SCALE_FACTOR} to {ROOTDIR}')
    _create_tpcds_queries()

if __name__ == '__main__':
    assert BENCH_TYPE == 'tpc-h' or BENCH_TYPE == 'tpc-ds', 'unsupported benchmark type'
    if BENCH_TYPE == 'tpc-h':
        tpch()
    else:
        tpcds()