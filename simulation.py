#! /usr/bin/env python

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


'''
The main Soapy Simulation module

This module contains the ``Sim`` class,
which can be used to run an end-to-end
simulation. Initally, a configuration
file is read, the system is initialised,
interaction and command matrices
calculated and finally a loop run.
The simulation outputs some information
to the console during the simulation.

The ``Sim`` class holds all configuration
information and data from the simulation.

Examples:

    To initialise the class::

        import soapy
        sim = soapy.Sim("sh_8x8_4.2m.py")

Configuration information has now been loaded,
and can be accessed through the ``config``
attribute of the ``sim`` class. In fact,
each sub-module of the system has a
configuration object accessed through
this config attribute::

        print(sim.config.sim.pupilSize)
        sim.config.wfss[0].pxlsPerSubap = 10

Next, the system is initialised,
this entails calculating various
parameters in the system sub-modules,
so must be done after changing some
simulation parameters::

        sim.aoinit()

DM Interation and command matrices
are calculated now. If ``sim.config.sim.simName``
is not ``None``, then these matrices will be
saved in ``data/simName``
(data will be saved here also in a
time-stamped directory)::

        sim.makeIMat()


    Finally, the loop is run with the command::

        sim.aoloop()

    Some output will be printed to the console.
    After the loop has finished, data specified to
    be saved in the config file will be saved to
    ``data/simName`` (if it is not set to ``None``).
    Data can also be accessed from the simulation class,
    e.g. ``sim.allSlopes``, ``sim.longStrehl``


:Author:
    Andrew Reeves

'''

# standard python imports
import datetime
import os
import time
import traceback
from multiprocessing import Process, Queue
from argparse import ArgumentParser
import shutil
import importlib
import threading
from matplotlib import pyplot as plt
import numpy as np
import sys, os

sys.path.append(os.getcwd())


import numpy
#Use pyfits or astropy for fits file handling
try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("soapy requires either pyfits or astropy")

import aotools
import helper_functions as help_func

#sim imports
import atmosphere, logger, confParse, interp
import lineofsight

import shutil

#xrange now just "range" in python3.
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

class Sim(object):
    """
    The soapy Simulation class.

    This class holds all configuration
    information, data and control
    methods of the simulation.
    It contains high level methods dealing
    with initialising all component objects,
    making reconstructor control matrices,
    running the loop and saving data after
    the loop has run.

    Can be sub-classed and the 'aoloop' method
    overwritten for different loops to be used

    Args:
        configFile (string): The filename of
        the AO configuration file
    """

    def __init__(self, configFile=None):
        if not configFile:
            configFile = "conf/testConf.py"

        self.readParams(configFile)

        self.guiQueue = None
        self.go = False
        self._sim_running = False


    def readParams(self, configFile=None):
        """
        Reads configuration file parameters

        Calls the radParams function in confParse
        to read, parse and if required
        set reasonable defaults to AO parameters
        """

        if configFile:
            self.configFile = configFile

        logger.info("Loading configuration file...")
        self.config = confParse.loadSoapyConfig(self.configFile)
        logger.info("Loading configuration file... success!")


    def setLoggingLevel(self, level):

        """
        sets which messages are printed
        from logger.

        If logging level is set to 0,
        nothing is printed.
        
        If set to 1, only warnings are
        printed.

        If set to 2, warnings and info is
        printed.

        If set to 3 detailed debugging
        info is printed.

        parameters:
            level (int): the desired logging level
        """
        logger.setLoggingLevel(level)


    def aoinit(self):
        '''
        Initialises all simulation objects.

        Initialises and passes relevant data to sim objects. 
        This does important pre-run tasks, such as creating or 
        loading phase screens, determining WFS geometry, 
        setting propagation modes and pre-allocating data arrays 
        used later in the simulation.
        '''
        # Read params if they haven't been read before
        try:
            self.config.sim.pupilSize
        except:
            self.readParams()

        logger.setLoggingLevel(self.config.sim.verbosity)
        logger.setLoggingFile(self.config.sim.logfile)
        logger.info("Starting Sim: {}".format(self.getTimeStamp()))

        # Calculate some params from read ones
        self.config.calcParams()

        # Init Pupil Mask <-- TO INITIALIZE
        '''
        logger.info("Creating mask...")
        self.mask = make_mask(self.config)
        '''
        
        logger.info("Initializing atmosphere object...")
        self.atmos = atmosphere.atmos(self.config)
        
        logger.info("Initializing line of sight object...")
        self.los = lineofsight.LineOfSight( self.config,
                                            self.atmos.scrns)
        
        logger.info("Initializing SimHelper object...")
        self.SimHelper = help_func.SimHelper(self.config)

        self.initSaveData()
        
        #Initialize arrays for metric instantaneous metric calculation
        self.EField     = np.zeros(([   self.config.sim.simSize,
                                        self.config.sim.simSize]),
                                        dtype=complex)

        self.Intensity  = np.zeros(([   self.config.sim.simSize,
                                        self.config.sim.simSize])) 
        
        self.RX_Intensity  = np.zeros(([    self.config.sim.simSize,
                                            self.config.sim.simSize])) 

        # Init performance tracking
        self.iters  = 0
        
        self.Tlos   = 0
        self.Tatmos = 0
        self.Tsim   = 0

        logger.info("Initialisation Complete!")
    
    def loopFrame(self):
        """
        Runs a single from of the entire AO system.

        Moves the atmosphere, runs the WFSs, finds
        the corrective DM shape and finally runs 
        the science cameras. This can be called
        over and over to form the "loop"
        """
        #propagate field through the atmosphere
        t = time.time()
        self.EField[:] = self.los.frame()
        self.Tlos += time.time() - t
        
        #intro function pointer equivalent ? 
        self.Intensity[:] = self.SimHelper.calc_intensity(self.EField)
        self.RX_Intensity[:] = self.SimHelper.calc_RX_intensity()
        self.calc_inst_metrics()
        self.update_plots()
        
        # Get next phase screens
        t = time.time()
        self.atmos.moveScrns()
        self.Tatmos += time.time()-t

        # Run Loop...
        ########################################
        # Save Data
        # If sim is run continuously in loop,
        #overwrite oldest data in buffer
        
        self.storeData()
        #self.printOutput(self.iters, strehl=True)
        #self.addToGuiQueue()

        self.iters += 1
    
    def update_plots(self):
        '''
        If plot output is specified, plot instantanous metrics,
        phase screens
        '''
        
        if (self.config.sim.plotMetrics):
            pass
        '''
        self.SimHelper.plot_intensity(  self.config.rx.height,
                                        self.config.tel.telDiam,
                                        self.config.beam.propagationDir,
                                        self.iters)
        '''


    def calc_inst_metrics(self):
        '''
        Calculates instantenous power at RX and scintillation, 
        for immediate plotting
        '''
        #append is faster ?
        self.powerInstRX[self.iters]  = self.SimHelper.calc_RX_power()
        self.scintInstIdx[self.iters] = self.SimHelper.calc_scintillation_idx()
        
        #to normalize --> Interested in frequency variations
        if self.config.sim.saveSummedRXIntensityInTime:
                self.summedIntensity += self.RX_Intensity

    def aoloop(self):
        """
        Main AO Loop

        Runs a WFS iteration, reconstructs the phase,
        runs DMs and finally the science cameras.
        Also makes some nice output to the console
        and can add data to the Queue for the GUI if
        it has been requested. Repeats for nIters.
        """

        self.go = True
        try:
            while self.iters < self.config.sim.nIters:
                if self.go:
                    logger.info("{} iteration out of {}".format(self.iters+1,
                                                                self.config.sim.nIters))
                    self.loopFrame()
                else:
                    break
        except KeyboardInterrupt:
            self.go = False
            logger.info("\nSim exited by user\n")

        # Finally save data after loop is over.
        self.saveData()
        self.finishUp()


    def start_aoloop_thread(self):
        """
        Run the simulation continuously in a thread

        The Simulation will loop continuously as long as it is required. The data buffers for
        simulation are limited however to the size given by sim.config.nIters. Once this is 
        full, the oldest data will be overwritten.
        """
        if self._sim_running is False:
            self._loop_thread = threading.Thread(
                    target=self._aoloop_thread, daemon=True)

            self._sim_running = True
            self._loop_thread.start()


    def stop_aoloop_thread(self):
        """
        Stops the AO loop if its running continuously in a thread.

        Stops the simulation after the current iteration and joins the loop thread.
        Will save the data buffers to disk if configured to do so and v
        the output summary
        """

        if self._sim_running:
            # signal for thread to stop
            self._sim_running = False
            # wait for thread to finish
            self._loop_thread.join()

            # save data and finish up
            #self.saveData()
            self.finishUp()
            

    def _aoloop_thread(self):
        """
        Runs the AO Loop as a while loop to be used in a thread
        """
        while self._sim_running:
            self.loopFrame()


    def reset_loop(self):
        """
        Resets parameters in the system to zero, to restart an AO run wihtout reinitialising
        """
        self.iters = 0
                
        if self.config.sim.nSci > 0:
            self.longStrehl[:] = 0
            self.ee50d[:] = 0
            for sci in self.sciImgs.values(): sci[:] = 0
        
    def finishUp(self):
        
        """
        Prints a message to the console giving timing data. Used on sim end.
        """
        '''
        plt.imshow(self.summedIntensity)
        plt.show()

        plt.imshow(abs(np.fft.fft2(self.summedIntensity)))
        plt.show()
        '''

        logger.info("Power at RX: mean {:0.5f} (mW) std {:0.7f} (mW)".format(1e3*np.mean(self.powerInstRX),
                                                        1e3*np.std(self.powerInstRX)))

        logger.info("Scintillation idx at RX: mean {:0.5f}, std {:0.7f}".format(  np.mean(self.powerInstRX),
                                                                        np.std(self.powerInstRX)))
        logger.info("Time moving atmosphere: {:0.3f} (s)".format(self.Tatmos))
        logger.info("Time propagating Field through the atmoshpere: {:0.3f} (s)".format(self.Tlos))


    def initSaveData(self):
        '''
        Initialise data structures used for data saving.

        Initialise the data structures which will be used
        to store data which will be saved or analysed once
        the simulation has ended. If the ``simName = None``,
        no data is saved, other wise a directory called
        ``simName`` is created, and data from simulation runs
        are saved in a time-stamped directory inside this.
        '''
        logger.info("Initialise Data Storage...")
        
        # Initialise the FITS header to use. Store in `config.sim`
        self.config.sim.saveHeader = self.makeSaveHeader()
        if (self.config.sim.simName != None):

            self.path = self.config.sim.simName +"/"+self.timeStamp
            # make sure a different directory used by sleeping
            time.sleep(1)
            try:
                os.mkdir(self.path)
            except OSError:
                os.mkdir(self.config.sim.simName)
                os.mkdir(self.path)
            
            #Init saving metrics folders
            if self.config.sim.saveTotalIntensity:
                os.mkdir(self.path+"/TotalIntensity/")
                self.totalIntensity = np.zeros([self.config.sim.nIters,
                                                self.config.sim.simSize,
                                                self.config.sim.simSize])
            
            if self.config.sim.saveSummedRXIntensityInTime:
                os.mkdir(self.path+"/summedIntensity/")
                self.summedIntensity = np.zeros([self.config.sim.simSize,
                                                self.config.sim.simSize])

            if self.config.sim.saveTotalPower:
                os.mkdir(self.path+"/TotalPower/")
                self.totalPower = np.zeros(self.config.sim.nIters)

            if self.config.sim.saveRXIntensity:
                os.mkdir(self.path+"/RXIntensity/")
                self.totalRXIntensity = np.zeros([  self.config.sim.nIters,
                                                    self.config.sim.simSize,
                                                    self.config.sim.simSize])
            if self.config.sim.saveEField:
                os.mkdir(self.path+"/EField/")
                self.totalEField = np.zeros([   self.config.sim.nIters,
                                                self.config.sim.simSize,
                                                self.config.sim.simSize],
                                                dtype=np.complex64
                                                )
            if self.config.sim.saveRXPower:
                os.mkdir(self.path+"/RXPower/")
                self.powerInstRX  = numpy.zeros(self.config.sim.nIters)

            if self.config.sim.saveScintillationIndex:
                os.mkdir(self.path+"/ScintillationIndex/")
                self.scintInstIdx = numpy.zeros(self.config.sim.nIters)

            # Copy the config file to the save directory so you can 
            # remember what the parameters where
            if isinstance(self.config, confParse.YAML_Configurator):
                fname = "conf.yaml"
            else:
                fname = "conf.py"
            shutil.copyfile(self.configFile, os.path.join(self.path, fname))
        
        #self.ee50d = numpy.zeros((self.config.sim.nSci) ) <-- TO CHECK

    def storeData(self):
        """
        Stores data from each frame in an appropriate data structure.

        Called on each frame to store the simulation data into various data 
        structures corresponding to different data sources in the system.

        For some data streams that are very large, data gets saved to disk on 
        each iteration - this also happens here.

        Args:
            i (int): The system iteration number
        """
        #store the E FIELD from the LOS module ?

        if self.config.sim.saveTotalIntensity:
            self.totalIntensity[self.iters,:,:] = self.Intensity

        if self.config.sim.saveRXIntensity:
            '''To implement'''
            pass

        if self.config.sim.saveEField:
            self.totalEField[self.iters,:,:] =  self.EField

    def saveData(self):
        """
        Saves all recorded data to disk

        Called once simulation has ended to save the data recorded during 
        the simulation to disk in the directories created during initialisation.
        """
        logger.info("Saving data...") 
        # compute ee50d <-- TO CHECK
        '''
        for sci in range(self.config.sim.nSci):
            pxscale = self.sciCams[sci].fov / self.sciCams[sci].nx_pixels
            ee50d = aotools.encircled_energy(
                self.sciImgs[sci], fraction=0.5, eeDiameter=True) * pxscale
            if ee50d < (self.sciCams[sci].fov / 2):
                self.ee50d[sci] = ee50d
            else:
                logger.info(("\nEE50d computation invalid "
                             "due to small FoV of Science Camera {}\n").
                            format(sci))
                self.ee50d[sci] = None
        '''
            
        if self.config.sim.simName!=None:
            
            if self.config.sim.saveTotalIntensity:
                logger.info("Saving Total Intensity") 
                fits.writeto(self.path+"/TotalIntensity/TotalIntensity.fits",
                                self.totalIntensity,
                                header=self.config.sim.saveHeader,
                                overwrite=True)
            
            if self.config.sim.saveRXIntensity:
                logger.info("Saving RX Intensity") 
                pass
                '''
                fits.writeto(self.path+"/RXIntensity.fits",
                                self.totalRXIntensity,
                                header=self.config.sim.saveHeader,
                                overwrite=True)
                '''
            if self.config.sim.saveEField:
                #doesn't save complex data type !
                '''
                logger.info("Saving EField")
                fits.writeto(self.path+"EField/TotalIntensity.fits",
                            self.totalEField,
                            header=self.config.sim.saveHeader,
                            overwrite=True)
                '''    
    
    def makeSaveHeader(self):
        """
        Forms a header which can be used to give a header to FITS files saved by the simulation.
        """

        header = fits.Header()
        self.timeStamp = self.getTimeStamp()

        # Sim Params
        header["INSTRUME"] = "SOAPY"
        header["RTCNAME"] = "SOAPY"
        header["TELESCOP"] = "SOAPY"
        header["RUNID"] = self.config.sim.simName
        header["LOOP"] = True
        header["DATE-OBS"] = self.time.strftime("%Y-%m-%dT%H:%M:%S")
        header["NFRAMES"] = self.config.sim.nIters

        # Tel Params
        header["TELDIAM"] = self.config.tel.telDiam
        header["TELOBS"] = self.config.tel.obsDiam
        header["FR"] = 1./self.config.sim.loopTime
        
        # Atmos Params
        header["NBSCRNS"] = self.config.atmos.scrnNo
        header["SCRNALT"] = str(list(self.config.atmos.scrnHeights))
        header["SCRNSTR"] = str(list(self.config.atmos.scrnStrengths)) 
        header["WINDSPD"] = str(list(self.config.atmos.windSpeeds))
        header["WINDDIR"] = str(list(self.config.atmos.windDirs))

        #Beam
        header["POWER"]   = self.config.beam.power
        header["WVL"]     = self.config.beam.wavelength
        header["WAIST"]   = self.config.beam.beamWaist
        header["TYPE"]    = self.config.beam.type
        header["PROPDIR"] = self.config.beam.propagationDir

        #Receiver
        header["DIAMETER"] = self.config.rx.diameter
        header["HEIGHT"] = self.config.rx.height

        return header


    def getTimeStamp(self):
        """
        Returns a formatted timestamp

        Returns:
            string: nicely formatted timestamp of current time.
        """

        self.time = datetime.datetime.now()
        return self.time.strftime("%Y-%m-%d-%H-%M-%S")


    def printOutput(self, iter, strehl=False):
        """
        Prints simulation information  to the console

        Called on each iteration to print information about the current simulation, such as current strehl ratio, to the console. Still under development
        Args:
            label(str): Simulation Name
            iter(int): simulation frame number
            strehl(float, optional): current strehl ration if science cameras are present to record it.
        """
        if self.config.sim.simName:
            string = self.config.sim.simName.split("/")[-1]
        else:
            string = self.config.filename.split("/")[-1].split(".")[0]

        if strehl:
            string += "  Strehl -- "
            for sci in xrange(self.config.sim.nSci):
                string += "sci_{0}: inst {1:.2f}, long {2:.2f} ".format(
                        sci, self.sciCams[sci].instStrehl,
                        self.sciCams[sci].longExpStrehl)

        logger.statusMessage(iter, self.config.sim.nIters, string )


    def addToGuiQueue(self):
        """
        Adds data to a Queue object provided by the soapy GUI.

        The soapy GUI doesn't need to plot every frame from the simulation. 
        When it wants a frame, it will request if by setting 
        ``waitingPlot = True``. As this function is called on
        every iteration, data is passed to the GUI only if 
        ``waitingPlot = True``. 
        This allows efficient and abstracted interaction 
        between the GUI and the simulation
        """
        if self.guiQueue != None:
            if self.waitingPlot:
                guiPut = []
               
                sciImg = {}
                instSciImg = {}
                for i in xrange(self.config.sim.nSci):
                    try:
                        sciImg[i] = self.sciImgs[i].copy()
                    except AttributeError:
                        sciImg[i] = None
                    try:
                        instSciImg[i] = self.sciCams[i].detector.copy()
                    except AttributeError:
                        instSciImg[i] = None
                    
                guiPut = {  "wfsFocalPlane":wfsFocalPlane,
                            "sciImg":       sciImg,
                            "instSciImg":   instSciImg}

                self.guiLock.lock()
                try:
                    self.guiQueue.put_nowait(guiPut)
                except:
                    self.guiLock.unlock()
                    traceback.print_exc()
                self.guiLock.unlock()

                self.waitingPlot = False


def make_mask(config):
    """
    Generates a Soapy pupil mask

    Parameters:
        config (SoapyConfig): Config object describing Soapy simulation

    Returns:
        ndarray: 2-d pupil mask
    """
    if config.tel.mask == "circle":
        mask = aotools.circle(config.sim.pupilSize / 2.,
                                  config.sim.simSize)
        if config.tel.obsDiam != None:
            mask -= aotools.circle(
                config.tel.obsDiam * config.sim.pxlScale / 2.,
                config.sim.simSize)

    elif isinstance(config.tel.mask, str):
        maskHDUList = fits.open(config.tel.mask)
        mask = maskHDUList[0].data.copy()
        maskHDUList.close()
        logger.info('load mask "{}", of size: {}'.format(config.tel.mask, mask.shape))

        if not numpy.array_equal(mask.shape, (config.sim.pupilSize,) * 2):
            # interpolate mask to pupilSize if not that size already
            mask = numpy.round(interp.zoom(mask, config.sim.pupilSize))

    else:
        mask = config.tel.mask.copy()

    # Check its size is compatible. If its the pupil size, pad to sim size
    if (not numpy.array_equal(mask.shape, (config.sim.pupilSize,)*2)
            and not numpy.array_equal(mask.shape, (config.sim.simSize,)*2) ):
        raise ValueError("Mask Shape {} not compatible. Should be either `pupilSize` or `simSize`".format(mask.shape))

    if mask.shape != (config.sim.simSize, )*2:
        mask = numpy.pad(
                mask, config.sim.simPad, mode="constant")

    return mask

#######################
#Control Functions
######################
class DelayBuffer(list):
    '''
    A delay buffer.

    Each time delay() is called on the buffer, the input value is stored.
    If the buffer is larger than count, the oldest value is removed and returned.
    If the buffer is not yet full, a zero of similar shape as the last input
    is returned.
    '''

    def delay(self, value, count):
        self.append(value)
        if len(self) <= count:
            result = value*0.0
        else:
            for _ in range(len(self)-count):
                result = self.pop(0)
        return result


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("configFile",nargs="?",action="store")
    args = parser.parse_args()
    if args.configFile != None:
        confFile = args.configFile
    else:
        confFile = "conf/testConf.py"


    sim = Sim(confFile)
    print("AOInit...")
    sim.aoinit()
    sim.aoloop()
