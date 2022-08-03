import sys, os
sys.path.append(os.getcwd().replace("GUI",""))

import logger
import time

from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtWidgets import QApplication, QMainWindow

from fso_gui_ui import Ui_MainWindow

class GUI(QMainWindow):

    def __init__(self, sim = None, verbosity = None):
        QMainWindow.__init__(self)
            
        self.app = QCoreApplication.instance()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        #connect widgets

        self.ui.init_sim_pushButton.clicked.connect(self.init)
        self.ui.start_sim_pushButton.clicked.connect(self.start)
        self.ui.restart_sim_pushButton.clicked.connect(self.restart)
        self.ui.save_sim_pushButton.clicked.connect(self.save)
    
    def init(self):

        self.ui.sim_prog_label.setText("Initializing simulation loop...")

    def start(self):
            
        self.startTime = time.time()

        self.ui.sim_prog_label.setText("Running simulation loop...")

    def restart(self):
            
        self.startTime = time.time()

        self.ui.sim_prog_label.setText("Restarting simulation...")
    
    def save(self):
            
        self.startTime = time.time()

        self.ui.sim_prog_label.setText("Saving simulation...")

    def initPlots(self):
    

        pass

app = QApplication(sys.argv)
gui = GUI()
gui.show()
app.exec_()

