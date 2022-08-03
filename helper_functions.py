import sys, os

sys.path.append(os.getcwd())

import logger
from aotools import circle
from math import ceil
import matplotlib.pyplot as plt
import numpy as np


class SimHelper():
    
    """
    """
     
    def __init__(self, config):

        logger.info("Initializing SimHelper")
        
        self.config = config
        #beam parameters
        self.P0     = self.config.beam.power
        self.w0     = self.config.beam.beamWaist
        self.wvl    = self.config.beam.wavelength 
        
        #RX parameters
        self.N      =   self.config.sim.simSize
        self.dx     =   self.config.tel.telDiam/self.N
        self.n_rx_pxls = self.config.rx.diameter/self.dx   
       
        #Look into other libraries for correct calculations
        self.EField = 0
        self.Intensity = 0

    def calc_intensity(self,Efield):
        
        if (self.config.beam.type == 'gaussian'):
            
            I_0 = self.P0/(np.pi*self.w0**2)
            self.Intensity = I_0*np.abs(Efield)**2
            return self.Intensity 

        else:
            return None
            
    def calc_RX_intensity(self):
    
        """
        Calculates RX Intensity at the aperture 

        Returns:
            float : RX Intensity at aperture
        """

        aperture = circle(ceil(self.n_rx_pxls/2), self.N).astype(bool)
        
        return np.where(aperture,self.Intensity, 0)
    
    def calc_RX_power(self):
    
        """
        Calculates RX power at the aperture 

        Returns:
            float : RX power at aperture
        """       
        return self.dx**2*np.sum(self.calc_RX_intensity()) 

    def calc_scintillation_idx(self):

        """
        Calculates RX scintillation index 

        Returns:
            float : scintillation index 
        """
        I_aperture = self.calc_RX_intensity()
        return np.var(I_aperture)/np.mean(I_aperture)**2

    def gaussian_beam_ext(self, r_sq, z, flag = False):
    
        """
        Evaluates Gaussian beam field or its phase

        Parameters:
            r_sq (ndarray) : x**2 + y**2 (m)
            z    (float) : distance b/w TX and phase screen (m)
            flag (bool)  : returns phase if true, else returns Efield

        Returns:
            ndarray: calculated gaussian field at (xi,yi,zi)
        """

        #wave number
        k = 2*np.pi/self.wvl
    
        #refractive index of the medium
        n = 1 #to modify --> n = n(z)

        #Rayleigh range
        z_r = np.pi*self.w0**2*n/self.wvl

        #beam width
        w_z = self.w0*np.sqrt(1+(z/z_r)**2)

        #wavefront radius of curvature
        R_z = z*(1+(z_r/z)**2)

        #Guy phase
        ph_z = np.arctan(z/z_r)
    
        phase = np.exp(-1j*(k*z + k*r_sq/(2*R_z) - ph_z))

        if (flag):
            return phase
        return  self.w0/w_z*np.exp(-r_sq/w_z**2)*phase

    
    def plot_intensity(self, t):
   
       
        #determine extent
        L = self.config.tel.telDiam
        extent = -L/2, L/2, -L/2, L/2
    
        #figure, axes = plt.subplots(frameon = False)
        figure, axes = plt.subplots()

        #Z1 = np.add.outer(range(N), range(N)) % 2 
        #plt.imshow(Z1, interpolation='nearest', alpha = 0.9, extent=extent)
   
        plt.imshow(self.Intensity, alpha = .9, extent = extent)
    
        plt.xlabel(r'$x_n/2$' + ' (m)')
        plt.ylabel(r'$y_n/2$' + ' (m)')
    
        cmap = plt.colorbar()
        cmap.ax.set_xlabel(r'$W/m^2$')
        cmap.ax.set_ylabel(r'Intensity')

        plt.title(  self.config.beam.propagationDir +
                    'link gaussian beam intensity at ' + str(self.config.rx.height//1000) +  ' (km)')
        plt.clim(0,.1)

        plt.savefig('t' + str(t) + '.png')
        plt.close(figure)
        #plt.show()
