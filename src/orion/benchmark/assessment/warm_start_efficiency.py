from orion.benchmark.assessment.base import BaseAssess
from orion.plotting.base import regrets
from typing import List, Tuple
from orion.core.worker.experiment import Experiment
from plotly.graph_objects import Figure



class WarmStartEfficiency(BaseAssess):
    """
    TODO: Evaluate the 'warm start efficiency' (objective value) for each search algorithm
    at different time steps (trial number).
    """
    def __init__(self, task_num: int = 1):
        super().__init__(task_num=task_num)

    def analysis(self, task: str, experiments: List[Tuple[int, Experiment]]) -> Figure:
        """
        Generate a `plotly.graph_objects.Figure` to display the performance analysis
        based on the assessment purpose.

        task: str
            Name of the task
        experiments: list
            A list of (task_index, experiment), where task_index is the index of task to
            run for this assessment, and experiment is an instance of
            `orion.core.worker.experiment.Experiment`.
        """
        raise NotImplementedError(f"Add a cold/warm/hot-start plot")
