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


import  sys, os, traceback, time, queue, numpy as np

sys.path.append(os.getcwd().replace("GUI",""))

import simulation, logger, pyqtgraph as pg

from PyQt5 import (             QtWidgets,
                                QtCore)

from PyQt5.QtCore import (      Qt,
                                QCoreApplication,
                                QRunnable,
                                QThreadPool,
                                pyqtSlot,
                                QThread)

from PyQt5.QtWidgets import (   QApplication,
                                QMainWindow,
                                QFileDialog,
                                QLineEdit,
                                QSlider )

from PyQt5.QtGui import (       QDoubleValidator,
                                QValidator,
                                QWidget,
                                QVBoxLayout,
                                QSizePolicy)


from fso_gui_ui import Ui_MainWindow
from argparse import ArgumentParser

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERBOSITY   = 3
ACCEPTED    = QValidator.Acceptable
INT_TYPE    = 0
DBL_TYPE    = -1
MAX_SCI     = 100000


pg.graphicsItems.GradientEditorItem.Gradients = pg.pgcollections.OrderedDict(
        [
    ('viridis', {'ticks': [(0.,  ( 68,   1,  84, 255)),
                           (0.2, ( 65,  66, 134, 255)),
                           (0.4, ( 42, 118, 142, 255)),
                           (0.6, ( 32, 165, 133, 255)),
                           (0.8, (112, 206,  86, 255)),
                           (1.0, (241, 229,  28, 255))], 'mode':'rgb'}),
    ('coolwarm', {'ticks': [(0.0, ( 59,  76, 192)),
                            (0.5, (220, 220, 220)),
                            (1.0, (180, 4, 38))], 'mode': 'rgb'}),
    ('grey', {'ticks': [(0.0, (0, 0, 0, 255)),
                        (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}),
    ('magma', {'ticks':[(0., (0, 0, 3, 255)),
                        (0.25, (80, 18, 123, 255)),
                        (0.5, (182,  54, 121, 255)),
                        (0.75, (251, 136,  97, 255)),
                        (1.0, (251, 252, 191))], 'mode':'rgb'})
        ])


CMAP    =   {'mode': 'rgb','ticks':
            [   (0., (14, 66, 255, 255)),
                (0.5, (255, 255, 255, 255)),
                (1., (255, 26, 26, 255))]
            }

class GUI(QMainWindow):

    def __init__(self, sim = None, verbosity = None):
        QMainWindow.__init__(self)
        
        self.useOpenGL = False

        self.app = QCoreApplication.instance()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        #connect button widgets
        self.ui.init_sim_pushButton.clicked.connect(self.init)
        self.ui.start_sim_pushButton.clicked.connect(self.run)
        self.ui.stop_sim_pushButton.clicked.connect(self.stop)
        self.ui.restart_sim_pushButton.clicked.connect(self.restart)
        self.ui.save_sim_pushButton.clicked.connect(self.save)
        
        #connect menu widgets
        self.ui.load_params_action.triggered.connect(self.read_param_file)
        self.ui.reload_params_action.triggered.connect(self.reload_param_file)
        
        #initialize variables
        self.sim = sim
        self.config = self.sim.config
        self.output = self.ui.sim_prog_label.setText

        #initialize GUI input fields
        self.init_input_fields()
                
        #verify and validate user input <-- REDO ? NEEDS A SEPARATE THREAD ?
        self.verify_and_set_user_input()

        #define other variables
        self.loopRunning = False
        #Init Timer to update plots
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.setInterval(100)
        self.updateTimer.timeout.connect(self.update)
        #self.ui.updateTimeSpin.valueChanged.connect(self.updateTimeChanged)
        self.updateQueue = queue.Queue(10)
        self.updateLock = QtCore.QMutex()
        
        #init variables for sim threads
        self.initThread = None
        self.loopThread = None
        self.resultPlot = None
        
        #Required for plotting colors
        self.colorList = ["b","g","r","c","m","y","k"]
        self.colorNo = 0

        #self.resultPlot = PlotWidget()
        #self.ui.plotLayout.addWidget(self.resultPlot)
        
        #initialize GUI plots 
        #self.init_plots()
        self.init_metrics_plots() 

        #display GUI
        self.show()
        self.init()
    
    def moveEvent(self, event):
        """
        Overwrite PyQt Move event to force a repaint. (Might) fix a bug on some (my) macs
        """
        self.repaint()
        super(GUI, self).moveEvent(event)

    def verify_and_set_user_input(self):
        
        #Modify simulation logical window input fields
        #grid size
        self.ui.sim_grid_size_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.sim_grid_size_input, self.config.set_simSize, 0, 10000, INT_TYPE))
        #grid scale
        self.ui.sim_grid_scale_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.sim_grid_scale_input, self.config.set_gridScale, 0.1, 100, DBL_TYPE))
        #num iterations
        self.ui.sim_num_iter_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.sim_num_iter_input, self.config.set_nIters, 1, 10000, INT_TYPE))
        #sampling rate
        self.ui.sim_sample_rate_input.returnPressed.connect( lambda :
                self.validate_num(self.ui.sim_sample_rate_input, self.config.set_loopTime, 0, 10000, DBL_TYPE))
        
        #Modify optical beam logical window input fields

        #power
        self.ui.beam_power_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.beam_power_input, self.config.set_power, 0, 10000, DBL_TYPE))
        #wvl
        self.ui.beam_wvl_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.beam_wvl_input, self.config.set_wvl, 0, 10000, DBL_TYPE))
        #beam waist
        self.ui.beam_waist_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.beam_waist_input, self.config.set_beamWaist, 0, 10000, DBL_TYPE))
        #prop dir
        self.ui.beam_prop_dir_box.activated.connect(lambda :
                self.config.set_propagationDir(self.ui.beam_prop_dir_box.currentText())) 
        
        #Modify atmosphere logical window input fields
        
        #atmos scrn size
        self.ui.atmos_scrn_size_slider.valueChanged.connect(self.update_slider)
        #beam waist
        self.ui.atmos_n_scrn_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.atmos_n_scrn_input, self.config.set_scrnNo, 1, 10000, INT_TYPE))
        #Fried r0
        self.ui.atmos_fried_r0_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.atmos_fried_r0_input, self.config.set_r0, 0, 10000, DBL_TYPE))

        #wind speeds
        #To implmenet
        
        #wind dirs
        #To implmenet
        
        #Modify receiver logical window input fields
        
        #aperture diameter
        self.ui.rx_ap_diam_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.rx_ap_diam_input, self.config.set_diameter, 0, 10000, DBL_TYPE))
        #height
        self.ui.rx_height_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.rx_height_input, self.config.set_height, 0, 10000000, DBL_TYPE))
        #elevation
        self.ui.rx_elevation_input.returnPressed.connect( lambda : 
                self.validate_num(self.ui.rx_elevation_input, self.config.set_elevationAngle, 0, 180, DBL_TYPE))
        
        
    def update_slider(self):
        
        slider_value = 2**self.ui.atmos_scrn_size_slider.value()
        self.ui.atmos_scrn_size_label2.setText(str(slider_value))
        self.config.set_wholeScrnSize(slider_value)
    

    def validate_num(self, input_field, set_field, low, high, digits):

        if (digits == 0):
            num_type = int 
        else:
            num_type = float
        
        input_num = input_field.text().replace('.',',')

        validation_rule = QDoubleValidator(low, high, digits)
        validation_cond = validation_rule.validate(input_num, 0)[0] 
        
        if (validation_cond == ACCEPTED):

            self.output("Input is accepted !")
            set_field(num_type(input_num.replace(',','.'))) 
            input_field.setFocus()

        else:
            self.output("Input is invalid. Valid input is "         +   \
                                             str(num_type)          +   \
                                            " number ranging from " +   \
                                            str(low) +  " to "      +   \
                                            str(high) )

            input_field.setText('')

    
    def init_metrics_plots(self):
        
        self.power_result_plot = PlotWidget()
        self.scint_result_plot = PlotWidget()

        self.ui.power_rx_layout.addWidget(self.power_result_plot)
        self.power_rx_ax = self.power_result_plot.canvas.ax
        self.power_rx_ax.set_xlabel("Iterations",fontsize="medium")
        self.power_rx_ax.set_ylabel("Power (mW)",fontsize="medium")
        self.power_rx_ax.set_ylim(0, 2)
        self.power_rx_ax.set_xlim(0, self.config.sim.nIters)
        self.power_rx_ax.tick_params(axis='both', which='major', labelsize="xx-small")
        self.power_rx_ax.tick_params(axis='both', which='minor', labelsize="xx-small")
        
        #init scintillation_idx metrics
        self.ui.sci_idx_rx_layout.addWidget(self.scint_result_plot)
        
        self.sci_idx_ax = self.scint_result_plot.canvas.ax
        self.sci_idx_ax.set_xlabel("Iterations",fontsize="medium")
        self.sci_idx_ax.set_ylabel("Scintillation index ",fontsize="medium")
        self.sci_idx_ax.set_ylim(0, 1)
        self.sci_idx_ax.set_xlim(0, self.config.sim.nIters)
        self.sci_idx_ax.tick_params(axis='both', which='major', labelsize="xx-small")
        self.sci_idx_ax.tick_params(axis='both', which='minor', labelsize="xx-small")

        self.colorNo += 1
        if (self.colorNo == len(self.colorList)):
            self.colorNo = 0
       
            
    def update_metrics_plots(self):
        
        self.power_rx_ax.plot(  1e3*self.sim.powerInstRX[0:self.sim.iters],
                                ls = '-.',
                                color = self.colorList[(self.colorNo) % len(self.colorList)])
        self.sci_idx_ax.plot(   (MAX_SCI - self.sim.scintInstIdx[0:self.sim.iters])/MAX_SCI,
                                ls = '-.',
                                color = self.colorList[(self.colorNo) % len(self.colorList)])
       
        self.power_result_plot.canvas.draw()
        self.scint_result_plot.canvas.draw()

    def clear_metrics(self):
        
        try:
            self.power_rx_ax.clear()
            self.power_result_plot.canvas.draw()
        except AttributeError:
            logger.info("Power metric plot isn't cleared !")
        
        try:
            self.sci_idx_ax.clear()
            self.scint_result_plot.canvas.draw()
        except AttributeError:
            logger.info("Scintillation metric plot isn't cleared !")

        

    def init(self):
        
        self.ui.sim_prog_label.setText("Initializing from configuration file")
        self.ui.sim_prog_bar.setValue(1)
        
        self.iThread = InitThread(self)
        self.iThread.updateProgressSignal.connect(self.progressUpdate)
        self.iThread.finished.connect(self.init_plots)
        # self.iThread.finished.connect(self.plotPupilOverlap)
        self.iThread.start()
        self.config = self.sim.config
        

    def run(self):
        
        
        self.startTime = time.time()

        self.ui.sim_prog_label.setText("Running simulation loop...")
        self.ui.sim_prog_bar.setValue(0)
        
        self.loopThread = LoopThread(self)
        self.loopThread.updateProgressSignal.connect(self.progressUpdate)
        self.statsThread.updateStatsSignal.connect(self.updateStats)
        self.loopThread.start()

        self.updateTimer.start()
        self.statsThread.start()


    def stop(self):

        self.ui.sim_prog_label.setText("Stopping simulation loop...")
        self.sim.go = False

        try:
            self.loopThread.quit()
        except AttributeError:
            pass

        try:
            self.statsThread.quit()
        except AttributeError:
            pass
        self.updateTimer.stop()

    def restart(self):

        self.startTime = time.time()
        #reset sim variables
        self.sim.reset_loop()

        #reset plots
        self.clear_metrics()
        self.clear_plots()       
        self.ui.sim_prog_label.setText("Reset is complete...")

    
    def save(self):
            
        self.startTime = time.time()

        self.ui.sim_prog_label.setText("Saving simulation...")

    def init_input_fields(self):
        
        #simulation field
        self.ui.sim_grid_size_input.insert(str(self.config.sim.simSize))
        self.ui.sim_grid_scale_input.insert(str(self.config.tel.telDiam))
        self.ui.sim_num_iter_input.insert(str(self.config.sim.nIters))
        self.ui.sim_sample_rate_input.insert(str(self.config.sim.loopTime))
        
        #optical beam field
        self.ui.beam_power_input.insert(str(self.config.beam.power))
        self.ui.beam_wvl_input.insert(str(self.config.beam.wavelength))
        self.ui.beam_waist_input.insert(str(self.config.beam.beamWaist))
        self.ui.beam_prop_dir_box.setCurrentIndex(0)

        if (self.config.beam.propagationDir == 'up'):
            self.ui.beam_prop_dir_box.setCurrentIndex(0)
        else:
            self.ui.beam_prop_dir_box.setCurrentIndex(1)

        #atmosphere field
        scrn_size_log2 = int(np.log2(self.config.atmos.wholeScrnSize))
        self.ui.atmos_scrn_size_slider.setProperty("value",scrn_size_log2)
        self.ui.atmos_scrn_size_slider.setSliderPosition(scrn_size_log2)
        self.ui.atmos_scrn_size_label2.setText(str(self.config.atmos.wholeScrnSize))
        self.ui.atmos_n_scrn_input.insert(str(self.config.atmos.scrnNo))
        self.ui.atmos_wind_speed_input.insert(str(self.config.atmos.windSpeeds)[1:-1])
        self.ui.atmos_wind_dir_input.insert(str(self.config.atmos.windDirs)[1:-1])
        self.ui.atmos_fried_r0_input.insert(str(self.config.atmos.r0))
               
        self.ui.rx_ap_diam_input.insert(str(self.config.rx.diameter))
        self.ui.rx_height_input.insert(str(self.config.rx.height))
        self.ui.rx_elevation_input.insert(str(self.config.rx.elevationAngle))

    
    def read_param_file(self):

        fname = QFileDialog.getOpenFileName(self, 'Open file', '.')
        fname = str(fname[0])
        
        if fname:
            self.sim.readParams(fname)
            self.config = self.sim.config
            logger.info("Configuration file is read...")
            #self.init_plots()

    def reload_param_file(self):

        self.sim.readParams()

        
    def updateStats(self, itersPerSec, timeRemaining):

        self.ui.sim_prog_iters_label.setText(
                                "Iterations Per Second: %.2f"%(itersPerSec))
        self.ui.sim_prog_time_label.setText( "Time Remaining: %.2fs"%(timeRemaining) )
    #GUI callbacks

    def progressUpdate(self, message, i="", maxIter=""):

        if i!="" and maxIter!="":
            percent = int(round(100*(float(i)/float(maxIter))))
            self.ui.sim_prog_bar.setValue(percent)
            self.ui.sim_prog_labelsetText(
                    "{0}: Iteration {1} of {2}".format(message, i, maxIter))

        else:
            if i!="":
                message+=" {}".format(i)
            self.ui.sim_prog_label.setText(message)
    
    def updateTimeChanged(self):

        try:
            self.updateTime = int(numpy.round(1000./float(self.ui.updateTimeSpin.value())))
            self.updateTimer.setInterval(self.updateTime)
        except ZeroDivisionError:
            pass

    def update(self):

        #tell sim that gui wants a plot
        self.sim.waitingPlot = True
        #empty queue so only latest update is present
        plotDict = None
        self.updateLock.lock()
        try:
            while not self.updateQueue.empty():
                plotDict = self.updateQueue.get_nowait()
        except:
            self.updateLock.unlock()
            traceback.print_exc()
        self.updateLock.unlock()
        
        if plotDict:

            # Get the min and max plot scaling
            #scaleValues = self.getPlotScaling(plotDict)

            if (np.any(plotDict["Intensity_rx"]) != None):
                self.intensity_view.setImage(plotDict["Intensity_rx"])
            
            if (np.any(plotDict["Phase"]) != None):
                self.phase_view.setImage(plotDict["Phase"])


            if self.loopRunning:
                self.update_metrics_plots()
            self.app.processEvents()
    
    def getPlotScaling(self, plotDict):

        """
        Loops through all phase plots to find the required min and max values for plot scaling
        """
        plotMins = []
        plotMaxs = []
        for wfs in range(self.config.sim.nGS):
            if numpy.any(plotDict["wfsPhase"])!=None:
                plotMins.append(plotDict["wfsPhase"][wfs].min())
                plotMaxs.append(plotDict["wfsPhase"][wfs].max())

        for dm in range(self.config.sim.nDM):
            if numpy.any(plotDict["dmShape"][dm])!=None:
                plotMins.append(plotDict["dmShape"][dm].min())
                plotMaxs.append(plotDict["dmShape"][dm].max())

        for sci in range(self.config.sim.nSci):
            if numpy.any(plotDict["residual"][sci])!=None:
                plotMins.append(plotDict["residual"][sci].min())
                plotMaxs.append(plotDict["residual"][sci].max())

        # Now get the min and max of mins and maxs
        plotMin = min(plotMins)
        plotMax = max(plotMaxs)

        return plotMin, plotMax

    def makeImageItem(self, layout, size):

        gv = pg.GraphicsView()

        if self.useOpenGL and GL:
            gv.useOpenGL()
        
        layout.addWidget(gv)
        vb = pg.ViewBox()
        vb.setAspectLocked(True)
        vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

        gv.setCentralItem(vb)
        img = pg.ImageItem(border="w")
        vb.addItem(img)
        vb.setRange(QtCore.QRectF(0, 0, size, size))
        
        return img
    
    def init_plots(self):
        
        self.intensity_view = self.makeImageItem(   self.ui.intensity_layout,
                                                    self.sim.config.simSize)

        self.phase_view     = self.makeImageItem(   self.ui.phase_layout,
                                                    self.sim.config.simSize)
        
        self.sim.guiQueue = self.updateQueue
        self.sim.guiLock = self.updateLock
        self.sim.gui = True
        self.sim.waitingPlot = False
        
        self.ui.sim_prog_bar.setValue(100)
        self.statsThread = StatsThread(self.sim) 
        logger.info("Init plots is complete")
    
    def clear_plots(self):

        self.intensity_view.clear()
        self.phase_view.clear()

        
class StatsThread(QtCore.QThread):
    
    updateStatsSignal = QtCore.pyqtSignal(float,float)
    def __init__(self, sim):
        QtCore.QThread.__init__(self)

        self.sim = sim

    def run(self):
        self.startTime = time.time()

        while self.sim.iters+1 < self.sim.config.sim.nIters and self.sim.go:
            time.sleep(0.2)
            iTime = time.time()
            # try:
            #Calculate and print running stats
            itersPerSec = self.sim.iters / (iTime - self.startTime)
            if itersPerSec == 0:
                itersPerSec = 0.00001
            timeRemaining = (self.sim.config.sim.nIters-self.sim.iters)/itersPerSec
            self.updateStatsSignal.emit(itersPerSec, timeRemaining)

class InitThread(QtCore.QThread):

    updateProgressSignal = QtCore.pyqtSignal(str,str,str)
    init_done_signal = QtCore.pyqtSignal()
   
    def __init__(self, guiObj):
        QtCore.QThread.__init__(self)
        self.guiObj = guiObj
        self.sim = guiObj.sim

    def run(self):
        logger.setStatusFunc(self.progressUpdate)
        if self.sim.go:
            self.guiObj.stop()

        self.sim.aoinit()

    def progressUpdate(self, message, i="", maxIter=""):
        self.updateProgressSignal.emit(str(message), str(i), str(maxIter))

class LoopThread(QtCore.QThread):

    updateProgressSignal = QtCore.pyqtSignal(str,str,str)

    def __init__(self, guiObj):

        QtCore.QThread.__init__(self)
        #multiprocessing.Process.__init__(self)
        self.guiObj = guiObj

        self.sim = guiObj.sim

    def run(self):

        logger.setStatusFunc(self.progressUpdate)
        try:

            self.guiObj.loopRunning = True
            self.sim.aoloop()
            self.guiObj.loopRunning = False
            self.guiObj.stop()
       
        except:

            self.sim.go = False
            self.guiObj.loopRunning = False
            self.guiObj.stop()
            traceback.print_exc()


    def progressUpdate(self, message, i="", maxIter=""):

        self.updateProgressSignal.emit(str(message), str(i), str(maxIter))

class PlotCanvas(FigureCanvas):

    def __init__(self):
        self.fig = Figure(facecolor="white", frameon=False)
        self.ax = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class PlotWidget(QWidget):

    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.canvas = PlotCanvas()
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)


def start_gui(sim, verbosity=3):
    
    app = QtWidgets.QApplication([])

    gui = GUI(sim, verbosity=verbosity)
    
    app.exec_()

if (__name__ == '__main__'):
    
    parser = ArgumentParser()
    parser.add_argument("configFile",nargs="?",action="store")
    args = parser.parse_args()

    if args.configFile != None:
        confFile = args.configFile
    else:
        confFile = os.getcwd().replace("GUI","") + "run_sim/sim_conf.yaml"
    
    logger.setLoggingLevel(VERBOSITY)
    sim = simulation.Sim(confFile)
    start_gui(sim)
