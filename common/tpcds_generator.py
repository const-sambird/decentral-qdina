import glob
import os
import random
import shutil
import subprocess

from common.generator import WorkloadGenerator

class TPCDSGenerator(WorkloadGenerator):
    def __init__(self, template_path: str, queries_per_template: int | list[int], workload_path: str, scale_factor: int):
        '''
        Query generator. Invokes TPC-H `qgen` to create a number of queries to form a workload.

        :param qgen_path: the directory where dbgen/qgen is located. if not compiled, will be compiled.
        :param template_path: where the (potentially corrected) TPC-H templates, or other templates, are
        :param queries_per_template: how many queries to produce from each template? either an int (for a uniform distribution) or list of ints.
        :param workload_path: the output directory for the generated queries
        :param scale_factor: the scale factor for query generation
        '''
        rootdir = os.path.dirname(os.path.realpath(__file__))
        self.qgen_path = os.path.normpath(os.path.join(rootdir, './tpc-ds/tools'))
        self.template_path = os.path.normpath(os.path.join(rootdir, template_path, 'tpc-ds'))
        self.workload_path = os.path.normpath(os.path.join(rootdir, workload_path))
        self.num_templates = 99
        self.scale_factor = str(scale_factor)
        if isinstance(queries_per_template, list):
            self.queries_per_template = queries_per_template
        else:
            self.queries_per_template = [queries_per_template for _ in range(self.num_templates)]
    
    def create_queries(self):
        '''
        Create the queries, using the parameters given in the constructor.
        '''
        os.makedirs(self.workload_path, exist_ok=True)

        self._compile_dsdgen()
        # self._move_query_templates()
        self._create_queries()
    
    def get_workload(self) -> tuple[list[str], list[int]]:
        '''
        Reads the generated queries into memory.

        :returns queries: a list of every query in the workload
        :returns templates: which template each query is from
        '''
        with open(f'{self.workload_path}/templates.csv', 'r') as infile:
            templates = [int(x) for x in infile.read().split(',')]
        
        queries = []

        for i in range(self.num_templates):
            for j in range(self.queries_per_template[i]):
                with open(f'{self.workload_path}/{i + 1}_{j}.sql', 'r') as infile:
                    lines = infile.readlines()
                    flattened = ' '.join(lines[1:])
                    queries.append(flattened.replace('\n', ' ').replace('\t', ' '))
        
        return queries, templates
    
    def _compile_dsdgen(self):
        subprocess.run(['make', 'CC="gcc-9"'], cwd=self.qgen_path)
    
    def _move_query_templates(self):
        existing_templates = glob.glob(f'{self.qgen_path}/queries/*.sql')
        corrected_templates = glob.glob(f'{self.template_path}/*.sql')

        for template in existing_templates:
            os.remove(template)
        
        for template in corrected_templates:
            shutil.copy(template, f'{self.qgen_path}/queries')
    
    def _create_queries(self):
        templates = []
        SEED_BASE = random.randint(1_000_000_000, 9_999_999_999)

        for i in range(self.num_templates):
            for j in range(self.queries_per_template[i]):
                templates.append(str(i))
                with open(f'{self.workload_path}/{i + 1}_{j}.sql', 'w') as outfile:
                    subprocess.run([f'{self.qgen_path}/dsqgen',
                                    '-SCALE', self.scale_factor,
                                    '-RNGSEED', str(SEED_BASE + j),
                                    '-TEMPLATE', f'query{i + 1}.tpl',
                                    '-DIALECT', 'netezza',
                                    '-DIRECTORY', os.path.normpath(os.path.join(self.qgen_path, '..', 'query_templates')),
                                    '-FILTER', 'Y'],
                                cwd=self.qgen_path,
                                #env=dict(os.environ, DSS_QUERY=f'{self.qgen_path}/queries'),
                                stdout=outfile)
        
        with open(f'{self.workload_path}/templates.csv', 'w') as outfile:
            outfile.write(','.join(templates))
