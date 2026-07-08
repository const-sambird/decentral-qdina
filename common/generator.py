class WorkloadGenerator:
    def __init__(self):
        pass

    def create_queries(self):
        '''
        Create the queries, using the parameters given in the constructor.
        '''
        raise NotImplementedError()
    
    def get_workload(self) -> tuple[list[str], list[int]]:
        '''
        Reads the generated queries into memory.

        :returns queries: a list of every query in the workload
        :returns templates: which template each query is from
        '''
        raise NotImplementedError()
    
