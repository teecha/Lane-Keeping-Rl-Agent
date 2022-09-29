""" Base class for Experiment parameters """

from copy import deepcopy

class ExperimentParams:
    '''
    Base class
    '''

    def __init__(self, expConfig, parameters):
        self.experiment_dict = dict()
        self.prameters = parameters

        for keys in parameters.keys():
            if keys not in expConfig.keys():
                raise KeyError("Experiment key %s un-defined."%(keys))
            self.experiment_dict[keys] = {}

            for p, _type_ in parameters[keys]:
                
                if p not in expConfig[keys]:
                    raise KeyError("Experiment parameter %s un-defined"%(p))
                
                if not type(expConfig[keys][p]) == _type_:
                    raise TypeError("Expected %s, got %s for parameter %s and key %s"%(_type_.__name__, type(expConfig[keys][p]).__name__, p, keys))
                
                self.experiment_dict[keys][p] = expConfig[keys][p]

    def get_paramters(self, key):
        if key not in self.experiment_dict.keys():
            raise KeyError("%s is not valid experiment key."%(key))
        return list(self.experiment_dict[key].keys())
    
    def get_exp_keys(self):
        return list(self.experiment_dict.keys())
    
    def get_parameter_value(self, key, parameter):
        if key not in self.experiment_dict.keys():
            raise KeyError("%s is not a valid experiment key."%(key))
        
        if parameter not in self.experiment_dict[key].keys():
            raise KeyError("%s is not a valid parameter for %s"%(parameter, key))
        return self.experiment_dict[key][parameter]

    def set_parameter_value(self, key, parameter, value):
        if key not in self.experiment_dict.keys():
            raise KeyError("%s is not a valid experiment key."%(key))
        
        if parameter not in self.experiment_dict[key].keys():
            raise KeyError("%s is not a valid parameter for %s."%(parameter, key))

        if type(value) is not type(self.experiment_dict[key][parameter]):
            raise TypeError("type doesn't match for parameter %s and key %s."%(parameter, key))
        self.experiment_dict[key][parameter] = value