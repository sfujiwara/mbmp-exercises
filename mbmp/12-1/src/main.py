import os
from google.protobuf import text_format, json_format
import numpy as np
import pulp
from .proto.config_pb2 import Problem
import yaml


def load_problem() -> Problem:

    with open('problem.yml', 'r') as f:
        config = yaml.load(f)

    problem = json_format.ParseDict(config, Problem())

    with open(os.path.join('outputs', 'problem.pbtxt'), 'w') as f:
        f.write(str(problem))

    return problem


def create_variables(name):

    xs = []

    for i in range(n_oil_groups):

        n_oils = len(problem.oil_groups[i].oils)
        b = np.empty([n_oils, n_periods], dtype=object)

        for j in range(n_oils):

            for k in range(n_periods):

                name_ = problem.oil_groups[i].oils[j].name
                b[j, k] = pulp.LpVariable(name=f'{name_}_{name}_{k+1}')

        xs.append(b)

    return xs


if __name__ == '__main__':
    problem = load_problem()

    n_periods = len(problem.oil_groups[0].oils[0].prices)
    n_oil_groups = len(problem.oil_groups)

    p = pulp.LpProblem(problem.name, pulp.LpMaximize)

    # Create variables.
    buy = create_variables('buy')
    use = create_variables('use')
    store = create_variables('store')

    # Objective for revenue.
    objective = pulp.lpSum(use)

    # Objective for cost.
    for i in range(n_oil_groups):
        n_oils = len(problem.oil_groups[i].oils)
        b = np.empty([n_oils, n_periods], dtype=object)
        for j in range(n_oils):
            for k in range(n_periods):
                price = problem.oil_groups[i].oils[j].prices[k]
                x = buy[i][j][k]
                objective -= x * price

    import IPython; IPython.embed()
