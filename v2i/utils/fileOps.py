import os
import yaml

def parseYaml(path):
    if os.path.exists(path):
        fileHandler = open(path, 'r')
        data = yaml.safe_load(fileHandler)
        fileHandler.close()
        return data
    else:
        raise ValueError("Invalid file path : %s"%(path))

def dumpYaml(path, data):
    if os.path.exists(path):
        fileHandler = open(path, 'w')
        yaml.dump(data, fileHandler, default_flow_style=False)
        fileHandler.close()
    else:
        raise ValueError("Invalid file path : %s"%(path))