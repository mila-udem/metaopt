from orion.benchmark.assessment.base import BaseAssess
from orion.plotting.base import regrets
from typing import List, Tuple, Dict
from orion.core.worker.experiment import Experiment
from orion.client import ExperimentClient
from plotly.graph_objects import Figure
from .averageresult import AverageResult, regrets
from .averagerank import AverageRank, rankings
from collections import defaultdict
import plotly.graph_objs as go

WarmStartExperimentsTuple = Tuple[ExperimentClient, ExperimentClient, ExperimentClient]


class WarmStartEfficiency(BaseAssess):
    """
    TODO: Evaluate the 'warm start efficiency' (objective value) for each search algorithm
    at different time steps (trial number).
    """

    def __init__(self, task_num: int = 1):
        super().__init__(task_num=task_num)

    def analysis(
        self,
        task: str,
        experiments: Dict[int, List[WarmStartExperimentsTuple]]
    ) -> Figure:
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

        # TODO: Reuse the figures form other Assessments, but group by cold/warm/hot in
        # addition to by algorithm
        all_plots = []
        for algo_index, list_of_exp_tuples in experiments.items():
            figure_experiments_dict = defaultdict(list)

            for repetition_index, experiment_tuple in enumerate(list_of_exp_tuples):
                for stage, exp in zip(["cold", "warm", "hot"], experiment_tuple):
                    algorithm_name = list(exp.configuration["algorithms"].keys())[0]
                    figure_experiments_dict[f"{algorithm_name}_{stage}"].append(exp)
                    # figure_experiments_dict[stage].append(exp)

            # TODO: Change the title somehow?
            algo_regrets_plot = regrets(figure_experiments_dict)
            all_plots.append(algo_regrets_plot)

            algo_rankings_plot = rankings(figure_experiments_dict)
            all_plots.append(algo_rankings_plot)

        return all_plots
