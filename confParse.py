#Copyright Durham University and Andrew Reeves
#2014

# This file is part of soapy.

#     soapy is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     soapy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with soapy.  If not, see <http://www.gnu.org/licenses/>.
"""
A module to generate configuration objects for Soapy,
given a parameter file.

This module defines a number of classes, which when
instantiated, create objects used to configure the
entire simulation, or just submodules.
All configuration objects are stored in the
``Configurator`` object which deals with
loading parameters from file, checking some
potential conflicts and using parameters to
calculate some other parameters used in
parts of the simulation.

The ``ConfigObj`` provides a base class used
by other module configuration objects, and
provides methods to read the parameters from
the dictionary read from file, and set defaults
if appropriate. Each other module in the system
has its own configuration object, and for components
such as wave-front sensors (WFSs), Deformable Mirrors
(DMs), Laser Guide Stars (LGSs) and Science Cameras, 
lists of the config objects for each component are
created.


"""

import numpy
import traceback
import copy
import sys, os

sys.path.append(os.getcwd())

import logger

# Check if can use yaml configuration style
try:
    import yaml
    YAML = True
except ImportError:
    logger.info("Can't import pyyaml. Can only use old python config style")
    YAML = False

# Attributes that can be contained in all configs
CONFIG_ATTRIBUTES = [
        'N',
            ]

RAD2ASEC = 206264.849159
ASEC2RAD = 1./RAD2ASEC

class ConfigurationError(Exception):
    pass


class PY_Configurator(object):
    """
    The configuration class holding all
    simulation configuration information

    This class is used to load the parameter
    dictionary from file, instantiate each
    configuration object and calculate some
    other parameters from the parameters given.

    The configuration file given to this class
    must contain a python dictionary, named
    ``simConfiguration``. This must contain
    other dictionaries for each sub-module
    of the system, ``Sim``, ``Atmosphere``,
    ``Telescope``, ``WFS``, ``LGS``, ``DM``,
    ``Science``. For the final 4 sub-dictionaries,
    each entry must be formatted as a list
    (or numpy array) where each value
    corresponds to that component.

    The number of components on the module will
    only depend on the number set in the ``Sim``
    dict. For example, if ``nGS`` is set to 2 in 
    ``Sim``, then in the ``WFS`` dict, each parameters
    must have at least 2 entries, e.g. ``subaps : [10,10]``.
    If the parameter has more than 2 entries, then
    only the first 2 will be noted and any others discarded.

    Descriptions of the available parameters for each
    sub-module are given in that that config
    classes documentation

    Args:
        filename (string): The name of the configuration file
    """

    def __init__(self, filename):
        self.filename = filename

        # placeholder for param objs
        #self.scis = []

        self.telDiam = 0
        self.simSize = 0

        self.sim = SimConfig()
        self.atmos = AtmosConfig()
        self.tel = TelConfig()
        self.beam = BeamConfig()
        self.rx = ReceiverConfig()

    def readfile(self):

        #Exec the config file, which should contain a dict ``simConfiguration``
        try:
            with open(self.filename) as file_:
                exec(file_.read(), globals())
        except:
            traceback.print_exc()
            raise ConfigurationError(
                    "Error loading config file: {}".format(self.filename))

        self.configDict = simConfiguration

    def loadSimParams(self):
        
        #TO FIX LOGGER MESSAGING HERE 
        self.readfile()

        logger.debug("\nLoad Sim Params...")
        self.sim.loadParams(self.configDict["Sim"])

        logger.debug("\nLoad Atmosphere Params...")
        self.atmos.loadParams(self.configDict["Atmosphere"])

        logger.debug("\nLoad Telescope Params...")
        self.tel.loadParams(self.configDict["Telescope"])
        
        logger.info("\nLoad Beam Params...")
        self.beam.loadParams(self.configDict["Beam"])
        
        logger.info("\nLoad Receiver Params...")
        self.rx.loadParams(self.configDict["Receiver"])

        self.calcParams()
        
        logger.info("===Simulation parameters===")
        logger.info("Propagtion directon : {}".format(self.rx.propagationDir))
        logger.info("Simulation type : {}".format(self.sim.simType))

    def calcParams(self):
        """
        Calculates some parameters from the configuration parameters.
        """

        # Run calcparams on each config object
        self.sim.calcParams()
        self.atmos.calcParams()
        self.tel.calcParams()
        self.beam.calcParams()
        self.rx.calcParams()
        
        
        #-----------------------IMPORTANT-------------------#
        self.sim.pxlScale = (float(self.sim.pupilSize)/
                self.tel.telDiam)
        
        self.telDiam = self.tel.telDiam
        # We oversize the pupil to what we'll call the "simulation size"
        simPadRatio = (self.sim.simOversize-1)/2.
        self.sim.simPad = max(int(round(self.sim.pupilSize*simPadRatio)), 1)
        self.sim.simSize = self.sim.pupilSize + 2 * self.sim.simPad
        self.simSize = self.sim.simSize
        
        self.sim.scrnSize = int(round(self.sim.simSize))

        # Make scrn_size even
        if self.sim.scrnSize % 2 != 0:
            self.sim.scrnSize += 1
        
        #Oversize the simulation for physical propagation
        #self.sim.scrnSize *= 2
        
        logger.info("Pixel Scale: {0:.2f} pxls/m".format(self.sim.pxlScale))
        logger.info("subScreenSize: {:d} simulation pixels".format(int(self.sim.scrnSize)))

        # If outer scale is None, set all to really big number. Will introduce bugs when we make
        # telescopes with diameter >1000000s of kilometres
        if self.atmos.L0 is None:
            self.atmos.L0 = []
            for scrn in range(self.atmos.scrnNo):
                self.atmos.L0.append(10e9)
        

    def __iter__(self):
        
        objs = {'Sim': dict(self.sim),
                'Atmosphere': dict(self.atmos),
                'Telescope': dict(self.tel),
                'Beam': dict(self.beam),
                'Receiver': dict(self.rx)
                }
        
        for configName, configObj in objs.items():
            yield configName, configObj


    def __len__(self):
        # Always have sim, atmos, tel, recon, DMs, WFSs,  and Scis
        return 6 #sim, atmos, tel, Scis, beam, rx 
    
    #simulation setters
    def set_simSize(self, x):
        self.simSize = x
    
    def set_gridScale(self, x):
        self.tel.telDiam = x
    
    def set_nIters(self, x):
        self.sim.nIters = x
    
    def set_loopTime(self, x):
        self.sim.loopTime = x
    
    def set_simType(self, x):
        self.sim.simType = x

    #optical beam setters 
    def set_power(self, x):
        self.beam.power = x
    
    def set_wvl(self, x):
        self.beam.wavelength = x
    
    def set_beamWaist(self, x):
        self.beam.beamWaist = x
    
    def set_propagationDir(self, x):
        self.beam.propagationDir = x
    
    #atmosphere params
    def set_wholeScrnSize(self, x):
        self.atmos.wholeScrnSize = x
    
    def set_scrnNo(self, x):
        self.atmos.scrnNo = x
    
    def set_r0(self, x):
        self.atmos.r0 = x
    
    def set_windDirs(self, x):
        self.atmos.windDirs = x 
    
    def set_windSpeeds(self, x):
        self.atmos.windSpeeds = numpy.array(x)
    
    def set_L0(self, x):
        self.atmos.L0 = x
    
    def set_scrnStrengths(self, x):
        self.atmos.scrnStrengths = x
    
    def set_scrnHeights(self, x):
        self.atmos.scrnHeights = x

    #rx params
    def set_diameter(self, x):
        self.rx.diameter = x
    
    def set_height(self, x):
        self.rx.height = x
    
    def set_elevationAngle(self, x):
        self.rx.elevationAngle = x*numpy.pi/180
    
    def set_orbitalAltitude(self,x):

        self.rx.orbitalAltitude = x

class YAML_Configurator(PY_Configurator):

    def readfile(self):

        # load config file from Yaml file
        with open(self.filename) as file_:
            self.configDict = yaml.load(file_, Loader=yaml.SafeLoader)


    def loadSimParams(self):

        self.readfile()
        logger.info("FSO-TOOL")
        logger.debug("\nLoad Sim Params...")
        try:
            self.sim.loadParams(self.configDict["Sim"])
        except KeyError:
            self.sim.loadParams(self.configDict)

        logger.debug("\nLoad Atmosphere Params...")
        self.atmos.loadParams(self.configDict["Atmosphere"])

        logger.debug("\nLoad Telescope Params...")
        self.tel.loadParams(self.configDict["Telescope"])

        logger.info("\nLoad Beam Params...")
        self.beam.loadParams(self.configDict["Beam"])
        
        logger.info("\nLoad Receiver Params...")
        self.rx.loadParams(self.configDict["Receiver"])

        self.calcParams()


class ConfigObj(object):
    # Parameters that can be had by any configuration object

    def __init__(self, N=None):
        # This is the index of some config object, i.e. WFS 1, 2, 3..N
        self.N = N

    def warnAndExit(self, param):

        message = "{0} not set!".format(param)
        logger.warning(message)
        raise ConfigurationError(message)

    def warnAndDefault(self, param, newValue):
        #to make less noisy
        #message = "{0} not set, default to {1}".format(param, newValue)
        self.__setattr__(param, newValue)

        #logger.debug(message)
    def initParams(self):
        for param in self.requiredParams:
            self.__setattr__(param, None)

    def loadParams(self, configDict):

        if self.N!=None:
            for param in self.requiredParams:
                try:
                    self.__setattr__(param, configDict[param][self.N])
                except KeyError:
                    self.warnAndExit(param)
                except IndexError:
                    raise ConfigurationError(
                                "Not enough values for {0}".format(param))
                except:
                    raise ConfigurationError(
                            "Failed while loading {0}. Check config file.".format(param))

            for param in self.optionalParams:
                try:
                    self.__setattr__(param[0], configDict[param[0]][self.N])
                except KeyError:
                    self.warnAndDefault(param[0], param[1]) 
                    pass
                except IndexError:
                    raise ConfigurationError(
                                "Not enough values for {0}".format(param))
                except:
                    raise ConfigurationError(
                            "Failed while loading {0}. Check config file.".format(param))
        else:
            for param in self.requiredParams:
                try:
                    self.__setattr__(param, configDict[param])
                except KeyError:
                    self.warnAndExit(param)
                except:
                    raise ConfigurationError(
                            "Failed while loading {0}. Check config file.".format(param))

            for param in self.optionalParams:
                try:
                    self.__setattr__(param[0], configDict[param[0]])
                except KeyError:
                    self.warnAndDefault(param[0], param[1])
                except:
                    raise ConfigurationError(
                            "Failed while loading {0}. Check config file.".format(param))

        self.calcParams()

    def calcParams(self):
        """
        Dummy method to be overidden if required
        """
        pass

    def __iter__(self):
        for param in self.requiredParams:
            key = param
            val = self.__dict__[param] 
            if isinstance(val, numpy.ndarray):
                val = val.tolist()
            yield key, val
                
        for param in self.optionalParams:
            key = param[0]
            val = self.__dict__[param[0]]
            if isinstance(val, numpy.ndarray):
                val = val.tolist()
            yield key, val
            
    def __len__(self):
        return len(self.requiredParams)+len(self.optionalParams)

    def __setattr__(self, name, value):
        if name in self.allowedAttrs:
            self.__dict__[name] = value
        else:
            raise ConfigurationError("'{}' Attribute not a configuration parameter".format(name))

    def __repr__(self):
        return str(dict(self))

    def __getitem__(self, item):
        return self.__getattribute__(item)


class SimConfig(ConfigObj):
    """
    Configuration parameters relevant for the entire simulation. These should be held at the beginning of the parameter file with no indendation.

    Required:
        =============   ===================
        **Parameter**   **Description**
        -------------   -------------------
        ``pupilSize``   int: Number of phase points across the simulation pupil
        ``nIters``      int: Number of iteration to run simulation
        ``loopTime``    float: Time between simulation frames (1/framerate)
        =============   ===================


    Optional:
        ==================  =================================   ===============
        **Parameter**       **Description**                         **Default**
        ------------------  ---------------------------------   ---------------
        ``nGS``             int: Number of Guide Stars and
                            WFS                                 ``0``
        ``nDM``             int: Number of deformable Mirrors   ``0``
        ``nSci``            int: Number of Science Cameras      ``0``
        ``reconstructor``   str: name of reconstructor
                            class to use. See
                            ``reconstructor`` module
                            for available reconstructors.       ``"MVM"``
        ``simName``         str: directory name to store
                            simulation data                     ``None``
        ``wfsMP``           bool: Each WFS uses its own
                            process                             ``False``
        ``verbosity``       int: debug output for the
                            simulation ranging from 0
                            (no-ouput) to 3 (all debug
                            output)                             ``2``
        ``logfile``         str: name of file to store
                            logging data,                       ``None``
        ``learnIters``      int: Number of `learn` iterations
                            for Learn & Apply reconstructor     ``0``
        ``learnAtmos``      str: if ``random``, then
                            random phase screens used for
                            `learn`                             ``random``
        ``simOversize``     float: The fraction to pad the
                            pupil size with to reduce edge
                            effects                             ``1.2``
        ``loopDelay``       int: loop delay in integer count
                            of ``loopTime``                     ``0``
        ``threads``         int: Number of threads to use
                            for multithreaded operations        ``1``
        ==================  =================================   ===============

    Data Saving (all default to False):
        ======================      ===================
        **Parameter**               **Description**
        ----------------------      -------------------
        ``saveSlopes``              Save all WFS slopes. Accessed from sim with
                                    ``sim.allSlopes``
        ``saveDmCommands``          Saves all DM Commands. Accessed from sim
                                    with ``sim.allDmCommands``
        ``saveWfsFrames``           Saves all WFS pixel data. Saves to disk a
                                    after every frame to avoid using too much
                                    memory
        ``saveStrehl``              Saves the science camera Strehl Ratio.
                                    Accessed from sim with ``sim.longStrehl``
                                    and ``sim.instStrehl``
        ``saveWfe``                 Saves the science camera wave front error.
                                    Accessed from sim with ``sim.WFE``.
        ``saveSciPsf``              Saves the science PSF.
        ``saveInstPsf``             Saves the instantenous science PSF.
        ``saveInstScieField``       Saves the instantaneous electric field at focal plane.
        ``saveSciRes``              Save Science residual phase
        ``saveCalib``               Copy calibration (IM, Rec) to save directory
                                    of simulation
        ======================      ===================

    """
    requiredParams = [  "pupilSize",
                        "nIters",
                        "loopTime",
                        ]

    optionalParams = [      ("nSci", 0),
                            ("simName", None),
                            ("savePhaseScreens", False),
                            ("saveTotalIntensity", False),
                            ("saveSummedRXIntensityInTime", False),
                            ("saveTotalPower", False),
                            ("saveRXIntensity", False),
                            ("saveEField", False),
                            ("saveRXPower", True),
                            ("saveScintillationIndex", True),
                            ("plotMetrics", False),
                            ("verbosity", 2),
                            ("logfile", None),
                            ("simOversize", 1.02),
                            ("loopDelay", 0),
                            ("simType", 'static')
                        ]

    # Parameters which may be set at some point and are allowed
    calculatedParams = [    'pxlScale',
                            'simPad',
                            'simSize',
                            'scrnSize',
                            'totalWfsData',
                            'totalActs',
                            'saveHeader',
                    ]


    allowedAttrs = copy.copy(
            requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])

class AtmosConfig(ConfigObj):
    """
    Configuration parameters characterising the atmosphere. These should be held in the ``Atmosphere`` group in the parameter file.

    Required:
        ==================      ===================
        **Parameter**           **Description**
        ------------------      -------------------
        ``scrnNo``              int: Number of turbulence layers
        ``scrnHeights``         list, int: Phase screen heights in metres
        ``scrnStrengths``       list, float: Relative layer scrnStrength
        ``windDirs``            list, float: Wind directions in degrees.
        ``windSpeeds``          list, float: Wind velocities in m/s
        ``r0``                  float: integrated  seeing strength
                                (metres at 500nm)

        ==================      ===================

    Optional:
        ==================  =================================   ===========
        **Parameter**       **Description**                     **Default**
        ------------------  ---------------------------------   -----------
        ``scrnNames``       list, string: filenames of phase
                            if loading from fits files. If
                            ``None`` will make new screens.     ``None``
        ``subHarmonics``    bool: Use sub-harmonic screen
                            generation algorithm for better
                            tip-tilt statistics - useful
                            for small phase screens.             ``False``
        ``L0``              list, float: Outer scale of each
                            layer. Kolmogorov turbulence if
                            ``None``.                           ``None``
        ``randomScrns``     bool: Use a random set of phase
                            phase screens for each loop
                            iteration?                          ``False``
        ``infinite``        bool: Use infinite phase screens?
                            warning: EXPERIMENTAL!              ``False``
        ``tau0``            float: Turbulence coherence time,
                            if set wind speeds are scaled.      ``None``
        ``wholeScrnSize``   int: Size of the phase screens 
                            to store in the ``atmosphere`` 
                            object. Required if large screens
                            used.                               ``None``
        ``randomSeed``      int: Seed for the random number
                            generator used to make phase
                            screens. If None, seed is random.   ``None``
        ==================  =================================   ===========
    """

    requiredParams = [ "scrnNo",
                        "scrnHeights",
                        "scrnStrengths",
                        "windDirs",
                        "windSpeeds",
                        "wvl"
                        ]

    optionalParams = [ ("scrnNames",None),
                        ("subHarmonics",False),
                        ("L0", None),
                        ("randomScrns", False),
                        ("tau0", None),
                        ("r0", None),
                        ("infinite", False),
                        ("wholeScrnSize", None),
                        # ("elevationAngle", 90),
                        ("randomSeed", None),
                        ("l0", 0.01)
                       ]

    # Parameters which may be set at some point and are allowed
    calculatedParams = [
                        'normScrnStrengths',
                        'r0'
                        ]
    allowedAttrs = copy.copy(requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])


    def calcParams(self):
        # Turn lists into numpy arrays
        self.scrnHeights = numpy.array(self.scrnHeights, float)
        self.scrnStrengths = numpy.array(self.scrnStrengths, float)
        self.windDirs = numpy.array(self.windDirs,float)
        self.windSpeeds = numpy.array(self.windSpeeds, float)
        if self.L0 is not None:
            self.L0 = numpy.array(self.L0)

        if (self.r0 == None):
            
            c2n = self.scrnStrengths
            dz  = self.scrnHeights
            wvl = float(self.wvl)
            r_0i = (0.423*(2*numpy.pi/wvl)**2*c2n*dz)**(-3/5)
            self.r0 = round(numpy.sum(r_0i**(-5/3))**(-3/5),4)
            
        
class TelConfig(ConfigObj):
    
    """
        Configuration parameters characterising the Telescope. These should be held in the ``Telescope`` group in the parameter file.

    Required:
        =============   ===================
        **Parameter**   **Description**
        -------------   -------------------
        ``telDiam``     float: Diameter of telescope pupil in metres
        =============   ===================

    Optional:
        ==================  =================================   ===========
        **Parameter**       **Description**                     **Default**
        ------------------  ---------------------------------   -----------
        ``obsDiam``         float: Diameter of central
                            obscuration                         ``0``
        ``mask``            str: Shape of pupil (only
                            accepts ``circle`` currently)       ``circle``
        ==================  =================================   ===========

    """


    requiredParams = [ "telDiam",
                            ]

    optionalParams = [ ("obsDiam", 0),
                        ("mask", "circle")
                        ]
    calculatedParams = [  ]

    allowedAttrs = copy.copy(requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])

class BeamConfig(ConfigObj):
    
    """
    Configuration parameters characterising transmitted beam.

    
    Required:
        ==================      ============================================
        **Parameter**           **Description**
        ------------------      --------------------------------------------
        ``power``               float: Initial power of a transmitted beam
                                in Watts (W)
                                
        ``wavelength``          float: The wavelength of the transmitted beam
                                in meters (m)

        ``beamWaist``            float: initial beam size (i.e. waist radius)
                                in millimeters (mm)

        ``type``                str: Type of a laser beam, by default gaussian
        ==================      ============================================

    """


    requiredParams = [  "power",
                        "wavelength",
                        "beamWaist",
                        "type",
                        "propagationDir"
                        ]

    calculatedParams = []
     
    optionalParams = []

    allowedAttrs = copy.copy(requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])
    
    #to make proper warnings if any of value is not set 
    def calcParams(self):
        # Set some parameters to correct type
        if (self.wavelength is None):
            raise ConfigurationError("Must supply wavelength for Beam")
        self.wavelength = float(self.wavelength)

        if (self.power is None):
            raise ConfigurationError("Must supply power for Beam")
        self.power = float(self.power)
        
        if (self.beamWaist is None):
            raise ConfigurationError("Must supply beamWaist for Beam")
        self.beamWaist = float(self.beamWaist)

        

class ReceiverConfig(ConfigObj):
    
    """
    Configuration parameters characterising transmitted beam.

    
    Required:
        ==================      ============================================
        **Parameter**           **Description**
        ------------------      --------------------------------------------
        ``diameter``            float: diameter of RX aperture in meters (m) 
                                
        ``height``              float: height of RX camera in meters (m)

        ==================      ============================================

   Optional:
        ==================== =================================   ===========
        **Parameter**        **Description**                     **Default**
        -------------------- ---------------------------------   -----------
        ``elevationAngle``     float:                                90
        ==================== =================================   ===========

    """
    
    requiredParams = [  "diameter",
                        "height"
                        ]

    calculatedParams = ["orbitalAltitude"]
     
    optionalParams = [("elevationAngle", 90)]

    allowedAttrs = copy.copy(requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])
    
    #to make proper warnings if any of value is not set 
    def calcParams(self):
        # Set some parameters to correct type
        if (self.diameter is None):
            raise ConfigurationError("Must supply diameter for Receiver")
        self.diameter = float(self.diameter)

        if (self.height is None):
            raise ConfigurationError("Must supply height for Receiver")
        self.height = float(self.height)
        self.elevationAngle = float(self.elevationAngle)
        
        alpha = self.elevationAngle*numpy.pi/180
        R_e = 6.371*1e6
        self.orbitalAltitude = ( (self.height + R_e)**2 - R_e**2*numpy.cos(alpha)**2 )**(1/2)\
                            - R_e*numpy.sin(alpha)


def loadSoapyConfig(configfile):

    # Find configfile extension
    file_ext = configfile.split('.')[-1]

    # If YAML use yaml configurator
    if file_ext=='yml' or file_ext=='yaml':
        if YAML:
            config = YAML_Configurator(configfile)
        else:
            raise ImportError("Requires pyyaml for YAML config file")

    # Otherwise, try and execute as python
    else:
        config = PY_Configurator(configfile)

    config.loadSimParams()

    return config

# compatability
Configurator = PY_Configurator

def test():
    C = Configurator("conf/testConfNew.py")
    C.readfile()
    C.loadSimParams()

    print("Test Passesd!")
    return 0


if __name__ == "__main__":
    test()
