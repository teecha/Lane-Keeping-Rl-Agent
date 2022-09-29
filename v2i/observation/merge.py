from PIL import Image
from copy import deepcopy
from v2i.observation.base import RGB_COLORS
from v2i.observation.base import Occupancy, vehicle, lidarSensor

LANE_WIDTH = 3.2

VEHICLE_WIDTH = 1.8
VEHICLE_LENGTH = 5.0

HIGHWAY_Y_MAX = 70.72
HIGHWAY_Y_MIN = 57.81

class twoinMergeOccupancy(Occupancy):

    def __init__(self,
                 view_length,
                 view_width,
                 scale=30):

        super().__init__(view_width, view_length, scale)
    
    
    def drawRectange(self,
                     p,
                     rec_length,
                     rec_width,
                     fill,
                     outline,
                     surface,
                     drawer,
                     angle=0):
                
        points = [(0,0),
                  (rec_length, 0),
                  (rec_length, rec_width),
                  (0, rec_width),]
        translated_points = [self.translate(self.rotate_point(pt, angle),  p) for pt in points]
        drawer.polygon(translated_points, fill=fill, outline=outline)
        #drawer.line([translated_points[0], (translated_points[0][0]+4, translated_points[0][1] + 4)], fill=RGB_COLORS['white'], width=10)
        
    
    def drawLine(self,
                 p1,
                 p2,
                 fill,
                 width,
                 surface,
                 drawer):
        drawer.line([p1, p2], fill=fill, width=width)
        
    def draw_roads(self, rlVeh):

        centre_point = (self.view_length/2,
                        self.view_width/2)
        
        # Update the postion to the centre of the vehicle
        veh = deepcopy(rlVeh)
        veh.postion['x'] += VEHICLE_LENGTH/2
        veh.postion['y'] += VEHICLE_WIDTH/2

        # Draw drivable region
        drivable_y_max = min(HIGHWAY_Y_MAX, veh.postion['y'] + self.view_width/2)
        drivable_y_min = max(HIGHWAY_Y_MIN, veh.postion['y'] - self.view_width/2)

        points = [0, self.view_width/2  + (drivable_y_max - veh.postion['y']),
                  self.view_length, self.view_width/2 + (drivable_y_max - veh.postion['y']),
                  self.view_length, self.view_width/2 - (veh.postion['y'] - drivable_y_min),
                  0, self.view_width/2 - (veh.postion['y'] - drivable_y_min)]
        
        # scale points
        for idx, _ in enumerate(points):
            points[idx] *= self.scale

        
        self.drawer.polygon(points,
                            fill=RGB_COLORS['black'])
        '''
        # Draw center lane
        self.drawLine(p1=(0 * self.scale, (centre_point[1]-(LANE_WIDTH/2)) * self.scale),
                      p2=(self.view_length * self.scale, (centre_point[1]-(LANE_WIDTH/2)) * self.scale),
                      fill=RGB_COLORS['white'],
                      width=0,
                      surface=self.surface,
                      drawer=self.drawer)
        
        self.drawLine(p1=(0 * self.scale, (centre_point[1]+(LANE_WIDTH/2)) * self.scale),
                      p2=(self.view_length * self.scale, (centre_point[1]+(LANE_WIDTH/2)) * self.scale),
                      fill=RGB_COLORS['white'],
                      width=0,
                      surface=self.surface,
                      drawer=self.drawer)
        
        numLanes = (centre_point[1] - (LANE_WIDTH/2))/LANE_WIDTH
        numLanes = int(numLanes)
        
        for lane in range(0, numLanes):
            if (centre_point[1]-(LANE_WIDTH/2) - ((lane+1) * LANE_WIDTH)) >= HIGHWAY_Y_MIN and (centre_point[1]-(LANE_WIDTH/2) - ((lane+1) * LANE_WIDTH)) <= HIGHWAY_Y_MAX:
                self.drawLine(p1=(0 * self.scale, (centre_point[1]-(LANE_WIDTH/2) - ((lane+1) * LANE_WIDTH)) * self.scale),
                        p2=(self.view_length * self.scale, (centre_point[1]-(LANE_WIDTH/2) - ((lane+1) * LANE_WIDTH)) * self.scale),
                        fill=RGB_COLORS['white'],
                        width=0,
                        surface=self.surface,
                        drawer=self.drawer)
            if (centre_point[1] + (LANE_WIDTH/2) + ((lane+1) * LANE_WIDTH)) >= HIGHWAY_Y_MIN and (centre_point[1]-(LANE_WIDTH/2) - ((lane+1) * LANE_WIDTH)) <= HIGHWAY_Y_MAX:
                self.drawLine(p1=(0 * self.scale, (centre_point[1] + (LANE_WIDTH/2) + ((lane+1) * LANE_WIDTH)) * self.scale),
                        p2=(self.view_length * self.scale, (centre_point[1] + (LANE_WIDTH/2) + ((lane+1) * LANE_WIDTH)) * self.scale),
                        fill=RGB_COLORS['white'],
                        width=0,
                        surface=self.surface,
                        drawer=self.drawer)
        '''

    def get_positon(self, kernel_api, veh_id):
        return kernel_api.kernel_api.vehicle.getPosition(veh_id)
    
    def get_angle(self, kernel_api, veh_id):
        return kernel_api.kernel_api.vehicle.getAngle(veh_id)
    
    def drawVehicles(self, vehicles):
        for veh in vehicles:
            if veh.vehType == 'ego':
                x = (self.view_length/2) - (VEHICLE_LENGTH/2)
                y = (self.view_width/2) - (VEHICLE_WIDTH/2)
                color = RGB_COLORS['yellow']
                #print(x,y)
            else:
                delta_x = veh.postion['x'] - vehicles[0].postion['x']
                delta_y = veh.postion['y'] - vehicles[0].postion['y']
                x = (self.view_length/2) - (VEHICLE_LENGTH/2) + delta_x
                y = (self.view_width/2) - (VEHICLE_WIDTH/2) + delta_y
                color = RGB_COLORS['red']
            
            # Draw vehicle
            self.drawRectange(p=(x * self.scale, y * self.scale),
                              rec_length=VEHICLE_LENGTH * self.scale,
                              rec_width=VEHICLE_WIDTH * self.scale,
                              fill=color,
                              outline=None,
                              surface=self.surface,
                              drawer=self.drawer,
                              angle=veh.angle)
    
    def get_valid_vehicles(self, kernel_api):
        
        # list of valid vehicles
        vehicles = []

        # Ego vehicle
        rl_x, rl_y = self.get_positon(kernel_api,  'rl_0')
        assert rl_x != -2**30 or rl_y != -2**30, "%.4f, %.4f"%(rl_x, rl_y)
        rl_angle = 0
        
        # Instantiate Ego-vehicle's lidar sensor
        egoLidar = lidarSensor()
        egoLidar.set_x_range(lower=rl_x - (VEHICLE_LENGTH/2) - self.view_length/2,
                             higher=rl_x - (VEHICLE_LENGTH/2) + VEHICLE_LENGTH + (self.view_length/2))
        egoLidar.set_y_range(lower=rl_y - (self.view_width/2) - (VEHICLE_WIDTH/2),
                             higher=rl_y + (self.view_width/2) + (VEHICLE_WIDTH/2))
        
        # Create ego-vehicle
        rlVeh = vehicle(rl_x, rl_y, 'ego', rl_angle)
        rlVeh.postion['x'] -= VEHICLE_LENGTH
        rlVeh.postion['y'] -= (VEHICLE_WIDTH/2)
        vehicles.append(rlVeh)

        # Look for non-ego vehicles which lie in sensor's range.
        rl_ids = set(['rl_0'])
        other_vehicle_ids = set(kernel_api.vehicle.get_ids()).difference(rl_ids)
        for otherVehID in other_vehicle_ids:
            v_x, v_y = self.get_positon(kernel_api, otherVehID)
            v_angle = self.get_angle(kernel_api, otherVehID)

            if v_x == -2**30 or v_y == -2**30:
                pass
            elif v_x >= egoLidar.xRange['lower'] and v_x <= egoLidar.xRange['higher'] and v_y >= egoLidar.yRange['lower'] and v_y <= egoLidar.yRange['higher']:
                edge = kernel_api.vehicle.get_edge(otherVehID)
                
                veh = vehicle(v_x, v_y, 'none', v_angle)
                
                if edge == 'merge_one':
                    veh.vehType = 'merge_one'
                    
                elif edge == 'merge_two':
                    veh.vehType = 'merge_two'
                    
                else:
                    veh.vehType = 'highway'
                
                veh.postion['x'] -= VEHICLE_LENGTH
                veh.angle = 90 - v_angle
                veh.postion['y'] -= VEHICLE_WIDTH/2
                vehicles.append(veh)
        
        return vehicles


    def update_surface(self, kernel_api):

        # Reset the surface background.
        self.surface = self.init_surface(self.view_width * self.scale,
                                         self.view_length * self.scale,
                                         RGB_COLORS['gray'])
        self.drawer = self.init_drawer(self.surface)
        
        # Get vehicles in view, first vehicle is always ego-vehicle.
        vehicles = self.get_valid_vehicles(kernel_api)

        # Draw roads
        self.draw_roads(vehicles[0])

        # Draw vehicles
        self.drawVehicles(vehicles)

        self.surface = self.surface.transpose(Image.FLIP_TOP_BOTTOM)

        # Draw Occupancy grids

        
    
