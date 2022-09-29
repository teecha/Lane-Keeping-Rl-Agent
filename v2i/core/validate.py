from copy import deepcopy
from v2i.core.params import DEFAULT_SCENE_PARAMS, DEFAULT_CAR_FOLLOWING_PARAMS, DEFAULT_CAR_FOLLOWING_GEN_PARAMS


class validateAndMergeConfig:

    def __init__(self, envConfig):
        self.envConfig = envConfig

    def mergeAll(self):
        return (self.mergeDict(self.envConfig['scene']['scenario'], self.envConfig['scene']['scenario-setting'], DEFAULT_SCENE_PARAMS), self.mergeDict(self.envConfig['car-following-parameters']['model'], self.envConfig['car-following-parameters']['model-parameters'], DEFAULT_CAR_FOLLOWING_PARAMS), self.mergeDict(self.envConfig['gen-following-parameters']['model'], self.envConfig['gen-following-parameters']['model-parameters'], DEFAULT_CAR_FOLLOWING_GEN_PARAMS))
    
    def mergeDict(self, head, parameters, rootDict):
        mergedDict = deepcopy(rootDict)
        for param in parameters:
            if parameters[param] is not None:
                mergedDict[head][param] = parameters[param]
        return mergedDict
    
    def validate(self):
        self.validateScene()
        self.validateCarFollowing()
        self.validateCarFollowingGen() 

    def validateScene(self):
        scenes = DEFAULT_SCENE_PARAMS.keys()
        assert self.envConfig['scene']['scenario'] in scenes, "Scenario : %s, is not implemented yet."%(self.envConfig['scene']['scenario'])
        for parameter in self.envConfig['scene']['scenario-setting']:
            if parameter not in DEFAULT_SCENE_PARAMS[self.envConfig['scene']['scenario']].keys():
                raise ValueError("%s parameter is not a valid parameter for %s scenario."%(parameter, self.envConfig['scene']['scenario']))
    
    def validateCarFollowingGen(self):
        controllers = DEFAULT_CAR_FOLLOWING_GEN_PARAMS.keys()
        assert self.envConfig['gen-following-parameters']['model'] in controllers, "Controller : %s is not implemented yet."%(self.envConfig['gen-following-parameters']['model'])
        for parameter in self.envConfig['gen-following-parameters']['model-parameters']:
            if parameter not in DEFAULT_CAR_FOLLOWING_GEN_PARAMS[self.envConfig['gen-following-parameters']['model']].keys():
                raise ValueError("%s parameter is not a valid parameter for %s controller."%(parameter, self.envConfig['gen-following-parameters']['model']))

    def validateCarFollowing(self):
        controllers = DEFAULT_CAR_FOLLOWING_PARAMS.keys()
        assert self.envConfig['car-following-parameters']['model'] in controllers, "Controller : %s is not implemented yet."%(self.envConfig['car-following-parameters']['model'])
        for parameter in self.envConfig['car-following-parameters']['model-parameters']:
            if parameter not in DEFAULT_CAR_FOLLOWING_PARAMS[self.envConfig['car-following-parameters']['model']].keys():
                raise ValueError("%s parameter is not a valid parameter for %s controller."%(parameter, self.envConfig['car-following-parameters']['model']))