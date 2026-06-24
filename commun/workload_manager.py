import random

class WorkloadManager:
    def __init__(self, workload: list[str], templates: list[int], execution_mode: str, fraction: int | None):
        '''
        Container class for the workload to be sent to the Router.

        Note this is different from the (static) workload sent to the preprocessor,
        because we still want to generate a full complement of index candidates,
        but in a low-training-data environment or a workload shift environment,
        we need to modify which queries are actually sent to each DBMS instance
        for use in computing the reinforcement learning agent's reward function.

        :param workload: a list of every query in the full generated workload
        :param templates: which template # each query in the workload belongs to
        :param execution_mode: how should the workload change?
        :param fraction: what proportion of the full templates will be used in training
        '''
        self._workload = workload
        self._templates = templates
        self._analytical_workload = []
        self._analytical_templates = []
        self._update_workload = []
        self._update_templates = []
        self._unique_update_temps = set()
        self._partial_workload = workload
        self._partial_templates = templates
        self._full_workload = workload
        self._full_templates = templates
        self._selection_weights = [2 for _ in templates]
        self._num_full_queries = len(workload)
        self._num_full_templates = len(list(set(templates)))
        self._exe_mode = execution_mode
        self._fraction = fraction

    def select_queries(self):
        '''
        In the low data and workload drift scenarioes, we need to
        select a fraction of the templates to be used in the training set.
        '''
        num_templates = round(self._num_full_queries * self._fraction)

        selected_queries = random.choices(self._full_workload, weights=self._selection_weights, k=num_templates)
        
        self._partial_workload = []
        self._partial_templates = []

        for i, query in enumerate(self._full_workload):
            if query in selected_queries:
                self._partial_workload.append(query)
                self._partial_templates.append(self._full_templates[i])
        
        self._workload = self._partial_workload
        self._templates = self._partial_templates
    
    def update_workload(self):
        '''
        If we are in a workload drift experiment, then we need to vary which templates
        are present in the overall workload. If we are not in a workload drift experiment,
        then this function is a no-op.
        '''
        if self._exe_mode != 'drift':
            return
        
        self.select_queries()

        # update the selection weights for the next episode (cause the workload to drift)
        queries_per_template = len(self._full_workload) // self._num_full_templates
        template_to_increase = random.randint(0, self._num_full_templates - 1)
        for i in range(queries_per_template * template_to_increase, queries_per_template * (template_to_increase + 1)):
            self._selection_weights[i] += 1
    
    def workload(self) -> list[str]:
        '''
        Returns the training set workload to be used by the router.

        :returns: the queries in the training set
        '''
        return self._workload
    
    def templates(self) -> list[int]:
        '''
        Returns which template each query in the training set is generated from.

        :returns: the template assignment to each query
        '''
        return self._templates

    def queries(self) -> tuple[list[str], list[int]]:
        return self._analytical_workload, self._analytical_templates
    
    def updates(self) -> tuple[list[str], list[int]]:
        return self._update_workload, self._update_templates
    
    def num_queries(self) -> int:
        '''
        Returns the number of queries in the training set.

        :returns: The size of the training set
        '''
        return len(self._workload)
    
    def num_templates(self) -> int:
        '''
        Returns the number of unique query templates used in the
        training set.

        :returns: the number of templates
        '''
        return len(list(set(self._templates)))
    
    def num_full_templates(self) -> int:
        '''
        Returns the number of unique query templates used in the full workload.

        :returns: the number of templates
        '''
        return self._num_full_templates
    
    def set_to_partial(self):
        '''
        If we have set the currently active workload/template set to
        the full workload (ie to generate a routing table), we can
        reset it back to the partial one without reselecting templates here.
        '''
        self._workload = self._partial_workload
        self._templates = self._partial_templates
    
    def set_to_full(self):
        '''
        Changes the active workload to the full set, rather than the partial.
        '''
        self._workload = self._full_workload
        self._templates = self._full_templates
    
    def partial_templates(self):
        '''
        Get which templates were used in the training set.

        :returns templates: which template numbers are used
        '''
        return sorted(list(set(self._partial_templates)))
    
    def sort(self):
        for i, statement in enumerate(self._workload):
            if 'select' in statement.lower():
                self._analytical_workload.append(statement)
                self._analytical_templates.append(self._templates[i])
            else:
                self._update_workload.append(statement)
                self._update_templates.append(self._templates[i])
                self._unique_update_temps.add(self._templates[i])
        print(len(self._analytical_workload), 'analytical queries;', len(self._update_workload), 'update queries')
    
    def update_templates(self):
        return self._unique_update_temps
