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

    # Write problem as pbtxt.
    with open(os.path.join('outputs', 'problem.pbtxt'), 'w') as f:
        f.write(str(problem))

    return problem


def create_variables(problem: Problem, name: str):

    if name not in ['USE', 'BUY', 'STORE']:
        raise ValueError

    n_oil_groups = len(problem.oil_groups)
    n_months = problem.n_months
    xs = []

    if name == 'STORE':
        n_months -= 1

    for i in range(n_oil_groups):

        n_oils = len(problem.oil_groups[i].oils)
        b = np.empty([n_oils, n_months], dtype=object)

        for j in range(n_oils):

            for k in range(n_months):

                name_ = problem.oil_groups[i].oils[j].name
                b[j, k] = pulp.LpVariable(name=f'{name_}_{name}_MONTH{k+1}', lowBound=0)

        xs.append(b)

    return xs


def main():

    problem: Problem = load_problem()

    n_months = len(problem.oil_groups[0].oils[0].prices)
    n_oil_groups = len(problem.oil_groups)

    model = pulp.LpProblem(problem.name, pulp.LpMaximize)

    # Create variables.
    buy = create_variables(problem, name='BUY')
    use = create_variables(problem, name='USE')
    store = create_variables(problem, name='STORE')

    # Objective for revenue.
    objective = pulp.lpSum(use) * problem.final_product_price

    # Objective for cost.
    for i in range(n_oil_groups):
        n_oils = len(problem.oil_groups[i].oils)
        for j in range(n_oils):
            for k in range(n_months):
                price = problem.oil_groups[i].oils[j].prices[k]
                x = buy[i][j][k]
                objective -= x * price

    # Objective for storage cost.
    objective -= pulp.lpSum(store) * problem.storage_cost

    # Constraints for refinement limit.
    for i in range(n_oil_groups):
        for j in range(n_months):
            constraint = pulp.lpSum(use[i][:, j]) <= problem.oil_groups[i].refinement_limit
            model.addConstraint(
                constraint=constraint,
                name=f'REFINEMENT_LIMIT_{problem.oil_groups[i].name}_MONTH{j+1}'
            )

    # Constraints for balance.
    for i in range(n_oil_groups):
        n_oils = len(problem.oil_groups[i].oils)
        for j in range(n_oils):
            for k in range(n_months):
                if k == 0:
                    constraint = 500 + buy[i][j][k] - use[i][j][k] - store[i][j][k] == 0
                elif k == n_months - 1:
                    # import IPython;IPython.embed()
                    constraint = store[i][j][k-1] + buy[i][j][k] - use[i][j][k] - 500 == 0
                else:
                    constraint = store[i][j][k-1] + buy[i][j][k] - use[i][j][k] - store[i][j][k] == 0
                model.addConstraint(
                    constraint=constraint,
                    name=f'BALANCE_{problem.oil_groups[i].oils[j].name}_MONTH{k+1}'
                )

    # Variables for total use.
    y = np.empty(shape=n_months, dtype=object)
    for m in range(n_months):
        y[m] = pulp.LpVariable(name=f'TOTAL_USE_MONTH{m+1}', lowBound=0)

    # Constraints for total use.
    for m in range(n_months):
        lhs = 0
        for og in range(n_oil_groups):
            lhs += pulp.lpSum(use[og][:, m])
        constraint = lhs == y[m]
        model.addConstraint(constraint, name=f'TOTAL_USE_MONTH{m+1}')

    # Constraints for hardness.
    for m in range(n_months):
        lhs = 0
        for og in range(n_oil_groups):
            n_oils = len(problem.oil_groups[og].oils)
            for o in range(n_oils):
                hardness = problem.oil_groups[og].oils[o].hardness
                lhs += hardness * use[og][o][m]

        constraint_u = lhs <= problem.hardness_upper_limit * y[m]
        constraint_l = lhs >= problem.hardness_lower_limit * y[m]
        model.addConstraint(constraint=constraint_u, name=f'HARNESS_UPPER_BOUND_MONTH{m+1}')
        model.addConstraint(constraint=constraint_l, name=f'HARNESS_LOWER_BOUND_MONTH{m+1}')

    model.setObjective(objective)
    model.writeLP('outputs/problem.lp')

    status = model.solve()

    print(f'Status: {pulp.LpStatus[status]}')

    for m in range(n_months):
        for og in range(n_oil_groups):
            n_oils = len(problem.oil_groups[og].oils)
            for o in range(n_oils):
                v_use = use[og][o, m]
                v_buy = buy[og][o, m]
                print(f'{v_use.name}: {v_use.value()}\t\t{v_buy.name}: {v_buy.value()}')

    # import IPython; IPython.embed()


if __name__ == '__main__':
    main()
