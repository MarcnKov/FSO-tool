"""
A generalised module to provide phase or the EField through a "Line Of Sight"

Line of Sight Object
====================
The module contains a 'lineOfSight' object,
which calculates the resulting phase or
complex amplitude from propogating through
the atmosphere in a given direction. This
can be done using either geometric
propagation, where phase is simply
summed for each layer, or physical
propagation, where the phase is
propagated between layers using
an angular spectrum propagation method.
Light can propogate either up or down.

The Object takes a 'config' as an
argument, which is likely to be
the same config object as the
module using it (WFSs, ScienceCams, or LGSs).
It should contain paramters required, such
as the observation direction and light
wavelength. The `config` also determines
whether to use physical or geometric
propagation through the 'propagationMode' parameter.

Examples::

    from soapy import confParse, lineofsight

    # Initialise a soapy conifuration file
    config = confParse.loadSoapyConfig('conf/sh_8x8.py')

    # Can make a 'LineOfSight' for WFSs
    los = lineofsight.LineOfSight(config.wfss[0], config)

    # Get resulting complex amplitude through line of sight
    EField = los.frame(some_phase_screens)

"""

import numpy
import numpy as np
import matplotlib.pyplot as plt

import aotools
from aotools import opticalpropagation

import sys, os
sys.path.append(os.getcwd())

import helper_functions as help_func
import logger, interp
import numbalib
import gc

DTYPE = numpy.float32
CDTYPE = numpy.complex64

# Python3 compatability
try:
    xrange
except NameError:
    xrange = range

RAD2ASEC = 206264.849159
ASEC2RAD = 1./RAD2ASEC

class LineOfSight(object):
    """
    A "Line of sight" through a number of turbulence
    layers in the atmosphere, observing ing a given
    direction.

    Parameters:
        config: The soapy config for the line of sight

        simConfig: The soapy simulation config object

        propagation_direction (str, optional):  Direction of
                                                light propagation,
                                                either `"up"` or
                                                `"down"`

        outPxlScale (float, optional):  The EField pixel
                                        scale required at
                                        the output (m/pxl)

        nOutPxls (int, optional):   Number of pixels
                                    to return in EFIeld

        mask (ndarray, optional):   Mask to apply at
                                    the *beginning*
                                    of propagation

        metaPupilPos (list, dict, optional):    A list or
                                                dictionary of
                                                the meta pupil
                                                position at each
                                                turbulence layer
                                                height ub metres.
                                                If None, works it
                                                out from GS position.
    """
    def __init__(   self,
                    soapyConfig,
                    phase_screens = None,
                    out_pixel_scale = None,
                    nx_out_pixels = None,
                    mask = None):

        self.soapy_config = soapyConfig
        
        self.pupil_size = self.soapy_config.sim.pupilSize
        self.phase_pixel_scale = 1./self.soapy_config.sim.pxlScale
        self.phase_screens = phase_screens
        
        self.sim_size = self.soapy_config.sim.simSize
        self.mask = mask 
        self.source_altitude = self.height
        self.nx_scrn_size = self.soapy_config.sim.scrnSize
        self.n_layers = self.soapy_config.atmos.scrnNo
        self.layer_altitudes = self.soapy_config.atmos.scrnHeights
        
        self.prop_dir   = self.soapy_config.beam.propagationDir
        self.power      = self.soapy_config.beam.power
        self.wvl        = self.soapy_config.beam.wavelength
        self.waist      = self.soapy_config.beam.beamWaist
        self.beam_type  = self.soapy_config.beam.type

        self.SimHelper = help_func.SimHelper(soapyConfig)

        self.calcInitParams(out_pixel_scale, nx_out_pixels)
        #self.allocDataArrays()

    # Some attributes for compatability between WFS and others
    @property
    def height(self):
        try:
            return self.soapy_config.rx.orbitalAltitude
        except AttributeError:
            return 1e3 
    '''
    @height.setter
    def height(self, height):
        try:
            self.config.height
            self.config.height = height
        except AttributeError:
            self.config.GSHeight
            self.config.GSHeight = height

    @property
    def position(self):
        try:
            return self.config.position
        except AttributeError:
            return self.config.GSPosition

    @position.setter
    def position(self, position):
        try:
            self.config.position
            self.config.position = position
        except AttributeError:
            self.config.GSPosition
            self.config.GSPosition = position
    '''

############################################################
# Initialisation routines


    def calcInitParams(self, out_pixel_scale=None, nx_out_pixels=None):
        """
        Calculates some parameters required later

        Parameters:
            outPxlScale (float): Pixel scale of required phase/EField (metres/pxl)
            nOutPxls (int): Size of output array in pixels
        """
        logger.debug("Calculate LOS Init PArams!")
        # Convert phase deviation to radians at wfs wavelength.
        # (currently in nm remember...?)
        #self.phs2Rad = 2*numpy.pi/(self.wavelength * 10**9)
        self.phs2Rad = 2*numpy.pi/(self.wvl)

        # Get the size of the phase required by the system
        self.in_pixel_scale = self.phase_pixel_scale

        if out_pixel_scale is None:
            self.out_pixel_scale = self.phase_pixel_scale
        else:
            self.out_pixel_scale = out_pixel_scale

        if nx_out_pixels is None:
            self.nx_out_pixels = self.soapy_config.sim.simSize
        else:
            self.nx_out_pixels = nx_out_pixels

        if (self.mask is None):
            self.mask = aotools.circle( self.pupil_size/2.,
                                        self.sim_size)
        else:
            self.mask = mask

        if self.mask is not None:
            self.outMask = interp.zoom( self.mask,
                                        self.nx_out_pixels)
        else:
            self.outMask = None
         
        if (self.phase_screens is not None):

            if (self.phase_screens.ndim == 2):
                self.phase_screens.shape = (1,
                                            self.phase_screens.shape[0],
                                            self.phase_screens.shape[1])
        # If no scrns, just assume no turbulence
        else:
            self.phase_screens = numpy.zeros(   (self.n_layers,
                                                self.nx_scrn_size,
                                                self.nx_scrn_size))

        #if propagation down, reverse order of phase screens
        if (self.prop_dir == 'down'):
            self.layer_altitudes    = self.layer_altitudes[::-1]
            self.phase_screens      = self.phase_screens[::-1]

        #FOCAL PLANE COORDINATE CALCULATION
        N = self.phase_screens[0].shape[0]
        #generate even spacing coordinates
        nx, ny =  np.meshgrid(  np.arange(-N/2,N/2),
                                np.arange(-N/2,N/2))
        
        #calculate spacing in the TX plane
        #to change
        self.in_pxl_scale  = self.out_pixel_scale 
        self.out_pxl_scale = self.out_pixel_scale 
        
        #fractional distance from 1 to i+1 plane
        alpha = self.layer_altitudes / self.layer_altitudes[-1]
        
        #grid spacing in the ith plane
        delta = (1-alpha)*self.in_pxl_scale + alpha*self.out_pxl_scale
        #observation plane coordinates
        xn = nx*delta[0]
        yn = ny*delta[0]
        
        self.r_sq = xn**2+yn**2 
        
        #for numerical stability reasons
        #apply absorbing boundaries to the E.Field
        w   = 0.47*N
        self.s_g = np.exp(-(nx**2+ny**2)**8/w**16)
        
        #Electric field
        #self.EField = np.ones([N] * 2, dtype=CDTYPE)

        self.EField = numpy.exp(
                1j*numpy.zeros((N,) * 2)).astype(CDTYPE)

        del xn, nx, yn, ny
        gc.collect()

    def allocDataArrays(self):
        """
        Allocate the data arrays the LOS will require

        Determines and allocates the various arrays the LOS will require to
        avoid having to re-alloc memory during the running of the LOS and
        keep it fast. This includes arrays for phase
        and the E-Field across the LOS
        """
                
        #Electric field
        self.EField = np.zeros([self.nx_out_pixels] * 2, dtype=CDTYPE)
        
        #turbuelnce phase screens
        self.phase_screens = np.zeros(      (self.n_layers,
                                           self.nx_out_pixels,
                                           self.nx_out_pixels))

######################################################

    def zeroData(self, **kwargs):

        """
        Sets the phase and complex amp data to zero
        """
        self.EField[:] = 1
        #self.phase_screens[:] = 0

    def physical_atmosphere_propagation(self):
        
        '''
        Finds total line of sight complex amplitude
        by propagating light through phase screens
        '''
        #SWTICH ON : SIMULATION VERIFICATION PURPOSES 
        #self.EField *= self.mask 
        
        self.s_g = 1 #<-- switch-off absorbing boundary

        phs2Rad = 2 * np.pi / self.wvl

        z_total  = 0 
        ht       = 0                 
        ht_final = self.soapy_config.rx.orbitalAltitude 
        
        # Propagate to first phase screen (if not already there)
        #TO CORRECT <-- WILL RESULT IN A BUG
        #CAN BE FURTHER OPTIMIZED --> MOVE TO CALCINITPARAMS
        if (ht != self.layer_altitudes[0]):
 
            if (self.prop_dir == "up"):
                z = abs(self.layer_altitudes[0] - ht)
            else:
                z = abs(ht_final - self.layer_altitudes[0])
            
            if (self.beam_type == 'gaussian'):
                self.EField = self.SimHelper.gaussian_beam_ext(self.r_sq, 0.01)
            
            self.EField[:] = opticalpropagation.angularSpectrum(    self.s_g*self.EField,
                                                                    self.wvl,
                                                                    self.in_pxl_scale,
                                                                    self.out_pxl_scale,
                                                                    z)
            z_total += z
            #self.EField[:] *= np.exp(-1j/(0.00001*self.wvl)*self.r_sq)

        #Propagate electrical field via phase screens
        for i in range(0, self.n_layers):
            
            #phase = 0

            phase = self.phase_screens[i]
             
            # Convert phase to radians
            phase *= phs2Rad
        
            # Change sign if propagating up
            if (self.prop_dir == 'up'):
                phase *= -1
            
            # Apply phase to EField
            self.EField *= np.exp(1j*phase)

            # Get propagation distance for the last layer
            if (i == self.n_layers-1):
                if (ht_final == 0):
                    # if the final height is infinity, don't propagate any more!
                    continue
                else:
                    z = abs(ht_final - ht) - z_total
            else:
                z = abs(self.layer_altitudes[i+1] - self.layer_altitudes[i])
            # Update total distance counter
            z_total += z
        
            # Do ASP for last layer to next
            self.EField[:] = opticalpropagation.angularSpectrum(    self.s_g*self.EField,
                                                                    self.wvl,
                                                                    self.in_pxl_scale,
                                                                    self.out_pxl_scale,
                                                                    z)
            #self.EField[:] *= np.exp(-1j/(0.00001*self.wvl)*self.r_sq)


    def frame(self):
        
        '''
        Runs one frame through a line of sight

        Finds the phase or complex amplitude through line of sight for a
        single simulation frame, with a given set of phase screens and
        some optional correction. 
        If scrns is ``None``, then light is propagated with no phase.

        Parameters:
            scrns (list): A list or dict containing the phase screens
            correction (ndarray, optional): The correction term to take
            from the phase screens before the WFS is run.
            read (bool, optional): Should the WFS be read out?
            if False, then WFS image is calculated but slopes not calculated.
            defaults to True.

        Returns:
            ndarray: WFS Measurements
        '''

        self.zeroData()
        
        self.physical_atmosphere_propagation()
        return self.EField
