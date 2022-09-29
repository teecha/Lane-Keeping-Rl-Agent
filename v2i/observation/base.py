""" Base class for creating occupancy grids. """

import cv2
import numpy as np
import skimage.measure
from warnings import warn
from collections import deque
from PIL import Image, ImageDraw



RGB_COLORS = {
              'black': (24,24,24),
              'white': (255, 255, 255),
              'green': (59, 196, 59),
              'red': (102,102,255),
              'gray': (128,128,128),
              'yellow': (153, 255, 255), }

class Occupancy:

    """
    Occupancy grid generator class.

    This class is used to generate the occupancy grid
    for a region defined using view-width and view-length.

    The base class is responsible for creating a drawer and surface
    to enable drawing the occupancy grid. The environment 
    specific drawing logic can be implemented using update_suraface
    method.

    Parameters
    ----------

        * view_width: The width of the occupancy grid in m.
        
        * view_length: The length of the occupany grid in m.
        
        * scale: The gui scalar, use to scale down or scale up the 
            the surface.
    """

    def __init__(self,
                 view_width,
                 view_length,
                 scale=30):
        """
        Intantiates the Occupancy class.

        Parameters
        ----------

        * view_width: The width of the occupancy grid in m.
        
        * view_length: The length of the occupany grid in m.
        
        * scale: The gui scalar, use to scale down or scale up the 
            the surface.
        
        """

        self.scale = scale
        self.view_length = view_length
        self.view_width = view_width

        # Create a surface to draw upon.
        self.surface = self.init_surface(view_width * scale,
                                         view_length * scale)
        
        # Create a drawer obeject for the surface.
        self.drawer = self.init_drawer(self.surface)
    
    def init_surface(self, width, length, fillColor=RGB_COLORS['black']):
        '''Creates a Pillow surface to draw shapes.

        Attributes
        ----------

        * width: The height of the surface in pixels.
        * length: The length of the surface in pixels.
        * fillColor: Suraface color.

        '''
        return Image.new('RGB',
                         (length, width),
                         fillColor)
    
    def init_drawer(self, surface):
        '''Returns a drawer object to draw on the surface.

        Attributes
        ----------

        * surface: A pillow surface to draw upon.
        '''

        return ImageDraw.Draw(surface)

    def get_occupancy(self, normalize=True):
        '''
        Convert the RGB occupancy of the ego-vehicle drawn
        on surface to Gray.
        '''
        image = self.surface.convert('L')
        image = np.array(image).copy()
        
        # Normalize grids
        if normalize:
            image = image.astype(np.float)
            image /= 255.0

        return image
    
    def rotate_point(self, p, angle):
        '''Rotate the point in 2D space specified via angle.

        Attributes
        ----------

        * p - Point to be rotated. Must be a 
                tuple of size 2.
        * angle - Size of the angle to be rotated
                by in degrees.
        '''
        cos_theta = np.cos(np.deg2rad(angle))
        sin_theta = np.sin(np.deg2rad(angle))
        return ((p[0]*cos_theta - p[1]* sin_theta),
                (p[0]*sin_theta + p[1] * cos_theta))
        
    def translate(self, p, offset):
        '''
        Translates the point specified using offset.

        Attributes
        ----------

        * p - Point to be translated. Must be
                a tuple of size 2.
        * offset - Translation offset. Must be
                a  tuple of size 2.
        '''

        return p[0] + offset[0], p[1] + offset[1]
    
    def render(self, ):
        '''Renders the current surface'''

        self.windowTitleRGB = 'Agent\'s occupancy Grid (RGB)'
        self.windowTitleGRAY = 'Agent\'s occupancy Grid (GRAY)'
        
        cv2.imshow(self.windowTitleRGB,
                    np.array(self.surface))
        
        occupancy = self.get_occupancy(normalize=False)
        
        cv2.imshow(self.windowTitleGRAY,
                    occupancy)
        
        cv2.waitKey(1)
    
    def __del__(self):
        try:
            cv2.destroyWindow(self.windowTitle)
        except Exception as e:
            warn("Failed to destroy render window.")

    def update_surface(self, kernel_api):

        '''
        Implement the logic to draw on the surface.

        This function is meant to provide an interace
        to the drawing surface using drawer object.
        '''

        raise NotImplementedError

class vehicle():
    '''
    Base class for holding vehicle state.

    Attributes:
    ----------

    * x : vehicle x-coordinate.
    * y : vehicle y-coordinate.
    * angle: vehicle's direction.
    * vehType: Type of the vehicle. Ego/Non-Ego Vehicle.
    
    '''
    def __init__(self, x, y, vehType, angle):
        self.postion = {'x': x,
                        'y': y}
        self.angle = angle
        self.vehType = vehType

class lidarSensor:
    '''
    Base class for lidar sensor.

    The class implements the methods and holds
    state of the ego vehicle\'s lidar sensor.
    '''

    def __init__(self):
        
        self.xRange = {'lower': None,
                       'higher': None}
        self.yRange = {'lower': None,
                        'higher': None}
    
    def set_x_range(self, lower, higher):
        '''
        Method to set the longitudinal range of the
        lidar sensor.

        Attributes
        ----------

        * lower: The lowest long. range of the lidar sensor.
        * higher: The highest long. range of the lidar sensor

        '''
        self.xRange['lower'] = lower
        self.xRange['higher'] = higher
    
    def set_y_range(self, lower, higher):
        '''
        Method to set the latitudinal range of the
        lidar sensor.

        Attributes
        ----------

        * lower: The lowest lat. range of the lidar sensor.
        * higher: The highest lat. range of the lidar sensor
        
        '''
        self.yRange['lower'] = lower
        self.yRange['higher'] = higher


class observationBuffer:

    """Base class for observation buffer.

    This class can be used to store the past k
    observations. The class offers convenient
    methods to reset or get the observations histories.

    Attributes
    ----------

    * memorySize: The number past observations to store.
    * expectedShape: The shape of the obsevation to expect.
    * initial_value: Default initial value for observations.
    """

    def __init__(self,
                 memorySize, 
                 expectedShape,
                 initial_value=0):

        self.memorySize = memorySize
        self.expectedShape = expectedShape
        self.initial_value = initial_value
        self.obsQueue = deque(maxlen=self.memorySize)
        self.reset()
    
    def reset(self):
        ''' Remove all the content from the buffer and 
            set its size to zero.
        '''
        self.obsQueue.clear()
        assert len(self.obsQueue) == 0
        for _ in range(self.memorySize):
            self.obsQueue.append(self.initial_value * np.ones(self.expectedShape))
    
    def addObs(self, obs):
        ''' Adds an observation to the buffer

        Attributes
        ----------

        * obs: Observation to be added in the buffer.
        '''
        shape = tuple(obs.shape)
        if shape != self.expectedShape:
            raise ValueError('Observation shape mis-match. Expected: ', self.expectedShape,', Got: ', shape)
            
        self.obsQueue.append(obs)
    
    def getObs(self):
        '''Returns the current state of the buffer.
        '''
        return np.stack(self.obsQueue, axis=0).copy()


if __name__ == "__main__":
    obj = Occupancy(view_width=10, view_length=10)
    obj.render()
    input("jbg")