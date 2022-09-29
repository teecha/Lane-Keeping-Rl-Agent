""" Base class for occupancy grids """

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

HIGHWAY_Y_MAX = 70.66
HIGHWAY_Y_MIN = 57.86
VEHICLE_LENGTH = 5


class sensorRange:

    def __init__(self, lower_range, higher_range):
        self.lower = lower_range
        self.higher = higher_range

class vehicle:

    def __init__(self, veh_id, x, y, lane, Vehtype, length, width):
        self.veh_id = veh_id
        self.position = [x, y]
        self.lane = lane
        self.type = Vehtype
        self.length = length
        self.width = width

class Occupancy:

    def __init__(self, local_view_size, lane_width, scale=30):
        self.local_view_size = local_view_size
        self.lane_width = lane_width
        self.scale = scale
        self.vecGetPositions = np.vectorize(self.get_position)
    
    def draw_roads(self, drawer, view_size, numLanes):
        # Draw road boundaries.
        start_height = 0
        drawer.line([(0, start_height+1), (view_size * self.scale, start_height+1)], fill=(0, 0, 0))
        for _ in range(0, numLanes-1):
            start_height += (self.lane_width * self.scale)
            drawer.line([(0, start_height), (view_size * self.scale, start_height)], fill=(0, 0, 0))
        start_height += (self.lane_width * self.scale)
        drawer.line([(0, start_height-2), (view_size * self.scale, start_height-2)], fill=(0, 0, 0))
    
    def get_position(self, vehID, vehicles):
        return vehicles.kernel_api.vehicle.getPosition(vehID)
    
    def vehicle_positions(self, k):
        # Get rl_vehicle id
        rl_ids = ['rl_0']
        if len(rl_ids) != 1:
            raise ValueError("Only single agent is supported in this scenario.")
        
        # Get rl vehicle position 
        rl_x, rl_y = self.get_position(rl_ids[0], self.k)
        assert rl_x != -2**30 or rl_y != -2**30    
        rl_lane = int(self.k.kernel_api.vehicle.getLaneID(rl_ids[0])[-1])
        assert rl_lane != ''

        # Ego vehicle's sensor range
        egoRange = sensorRange(rl_x -(VEHICLE_LENGTH/2) - self.local_view_size,
                                rl_x - (VEHICLE_LENGTH/2) + self.local_view_size + VEHICLE_LENGTH)

        # Build required rl_vehicle
        rl_vehicle = vehicle(rl_ids[0], 
                             rl_x, 
                             rl_y,
                             rl_lane, 
                             "rl", 
                             VEHICLE_LENGTH,
                             self.k.kernel_api.vehicle.getWidth(rl_ids[0]))
        
        rl_vehicle.position[0] -= rl_vehicle.length
        rl_vehicle.position[1] -= (rl_vehicle.width/2)

        # Get other vehicles positions
        other_vehicle_ids = set(self.k.vehicle.get_ids()).difference(rl_ids)
        valid_vehicles = []

        for other_veh_id in other_vehicle_ids:
            x,y = self.get_position(other_veh_id, k)
            
            if x == -2**30 or y == -2**30:
                print("warning, vehicle teleported.")
            elif x >= egoRange.lower and x <= egoRange.higher and y >= HIGHWAY_Y_MIN:
                lane = int(self.k.kernel_api.vehicle.getLaneID(other_veh_id)[-1])
                veh = vehicle(other_veh_id, x, y, lane, 'non-ego', VEHICLE_LENGTH, self.k.kernel_api.vehicle.getWidth(other_veh_id))
                veh.position[0] -= veh.length
                veh.position[1] -= (veh.width/2)
                valid_vehicles.append(veh)

        return valid_vehicles, rl_vehicle
    
    def draw_vehicles(self, drawer, veh, rlVeh):
        
        # Draw ego vehicle
        rlx = self.local_view_size - (VEHICLE_LENGTH/2)
        rly = rlVeh.position[1] - HIGHWAY_Y_MIN
        drawer.rectangle([(rlx * self.scale, rly * self.scale),
                         ((rlx + VEHICLE_LENGTH) * self.scale, (rly + rlVeh.width) * self.scale)], fill=(255, 0, 0), outline=(0,0,0), width=2)
        
        # Draw other vehicles
        for v in veh:
            delta_x = v.position[0] - rlVeh.position[0]
            other_x = self.local_view_size - (VEHICLE_LENGTH/2) + delta_x
            other_y = v.position[1] - HIGHWAY_Y_MIN
            drawer.rectangle([(other_x * self.scale, other_y * self.scale),
                         ((other_x + VEHICLE_LENGTH) * self.scale, (other_y + rlVeh.width) * self.scale)], fill=(0, 255, 0), outline=(0,0,0), width=2)

    def update(self, numLanes, vehicles_kernel):
        length = (2*self.local_view_size) * self.scale
        breadth = self.lane_width * self.scale
        breadth *= numLanes
        breadth = int(breadth)

        self.k = vehicles_kernel

        self.surface = Image.new('RGB', 
                                 (length, breadth),
                                 (255, 255, 255))

        self.drawer = ImageDraw.Draw(self.surface)
        self.draw_roads(self.drawer, 2*self.local_view_size, numLanes)


        otherVehicle, rlVehicle = self.vehicle_positions(vehicles_kernel)
        self.draw_vehicles(self.drawer, otherVehicle, rlVehicle)

        self.surface = self.surface.transpose(Image.FLIP_TOP_BOTTOM)
        cv2.imshow('frame', np.array(self.surface))
        cv2.waitKey(1)
        

