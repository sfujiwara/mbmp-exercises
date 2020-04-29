import pulp
from google.protobuf import text_format, json_format
from .proto.config_pb2 import *
import yaml


with open('config.yml', 'r') as f:
    config = yaml.load(f)

print(config)

problem = json_format.ParseDict(config, Problem())
print(problem)

import IPython; IPython.embed()
