from flow.core.params import VehicleParams
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.rlcontroller import RLController

from math import floor

class Vehicle:

    def __init__(self, numLanes, ringLength):
        self.maxVehiclesLimit = {
            'length': 230,
            'max_vehicles': 30
        }
        self.numLanes = numLanes
        self.ringLength = ringLength
    
    def get_number_vehicles(self, density, lanes, ringLength):
        if not (density >= 0.0 and density <= 1.0):
            raise ValueError("Density should between 0 and 1")
        singleLaneLimit = floor(ringLength * \
            (self.maxVehiclesLimit['max_vehicles']/self.maxVehiclesLimit['length']))
        return floor(density * lanes * singleLaneLimit)
    
    def add(self, density, rlVehicles):
        
        if rlVehicles < 0:
            raise ValueError("Number of RL vehicles should be atleast zero.")
        
        numVehicles = self.get_number_vehicles(density, self.numLanes, self.ringLength)
        numHumanVehicles = numVehicles - rlVehicles
        numHumanVehicles = max(0, numHumanVehicles)

        vehicles = VehicleParams()
        vehicles.add("human", acceleration_controller=(IDMController, {}), routing_controller=(ContinuousRouter, {}), num_vehicles=numHumanVehicles, color='red')
        vehicles.add("rl", acceleration_controller=(RLController, {}), routing_controller=(ContinuousRouter, {}), num_vehicles=rlVehicles, color="yellow")
        return vehicles
        