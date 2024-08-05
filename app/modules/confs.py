import json


def load_config():
    with open('../config.json') as f:
        config = json.load(f)
    return config
