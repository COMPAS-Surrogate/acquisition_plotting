__all__ = []


URL = "https://github.com/avivajpeyi/project_name"

__author__ = "Avi Vajpeyi "
__email__ = "avi.vajpeyi@gmail.com"
__uri__ = URL
__license__ = "MIT"
__description__ = "Project Description"
__copyright__ = "Copyright 2022 project_name developers"
__contributors__ = f"{URL}/graphs/contributors"


from .plot_objective import plot_objective
from .plot_evaluations import plot_evaluations
from .trieste import plot_trieste_objective
from .trieste import plot_trieste_evaluations