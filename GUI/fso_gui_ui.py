# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fso_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1122, 922)
        MainWindow.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.beam_power_label = QtWidgets.QLabel(self.centralwidget)
        self.beam_power_label.setGeometry(QtCore.QRect(20, 310, 171, 17))
        self.beam_power_label.setObjectName("beam_power_label")
        self.beam_wvl_label = QtWidgets.QLabel(self.centralwidget)
        self.beam_wvl_label.setGeometry(QtCore.QRect(20, 340, 181, 17))
        self.beam_wvl_label.setObjectName("beam_wvl_label")
        self.beam_waist_label = QtWidgets.QLabel(self.centralwidget)
        self.beam_waist_label.setGeometry(QtCore.QRect(20, 370, 171, 17))
        self.beam_waist_label.setObjectName("beam_waist_label")
        self.beam_prop_dir_label = QtWidgets.QLabel(self.centralwidget)
        self.beam_prop_dir_label.setGeometry(QtCore.QRect(20, 400, 141, 17))
        self.beam_prop_dir_label.setObjectName("beam_prop_dir_label")
        self.beam_prop_dir_box = QtWidgets.QComboBox(self.centralwidget)
        self.beam_prop_dir_box.setGeometry(QtCore.QRect(210, 400, 301, 25))
        self.beam_prop_dir_box.setObjectName("beam_prop_dir_box")
        self.beam_prop_dir_box.addItem("")
        self.beam_prop_dir_box.addItem("")
        self.sim_grid_size_slider = QtWidgets.QSlider(self.centralwidget)
        self.sim_grid_size_slider.setGeometry(QtCore.QRect(210, 120, 301, 51))
        self.sim_grid_size_slider.setOrientation(QtCore.Qt.Horizontal)
        self.sim_grid_size_slider.setObjectName("sim_grid_size_slider")
        self.sim_grid_size_label = QtWidgets.QLabel(self.centralwidget)
        self.sim_grid_size_label.setGeometry(QtCore.QRect(20, 140, 191, 17))
        self.sim_grid_size_label.setObjectName("sim_grid_size_label")
        self.sim_num_inter_label = QtWidgets.QLabel(self.centralwidget)
        self.sim_num_inter_label.setGeometry(QtCore.QRect(20, 200, 181, 17))
        self.sim_num_inter_label.setObjectName("sim_num_inter_label")
        self.sim_grid_scale_label = QtWidgets.QLabel(self.centralwidget)
        self.sim_grid_scale_label.setGeometry(QtCore.QRect(20, 170, 181, 17))
        self.sim_grid_scale_label.setObjectName("sim_grid_scale_label")
        self.sim_sample_rate_label = QtWidgets.QLabel(self.centralwidget)
        self.sim_sample_rate_label.setGeometry(QtCore.QRect(20, 230, 151, 17))
        self.sim_sample_rate_label.setObjectName("sim_sample_rate_label")
        self.rx_ap_diam_label = QtWidgets.QLabel(self.centralwidget)
        self.rx_ap_diam_label.setGeometry(QtCore.QRect(20, 670, 181, 17))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.rx_ap_diam_label.setFont(font)
        self.rx_ap_diam_label.setObjectName("rx_ap_diam_label")
        self.rx_elevation_label = QtWidgets.QLabel(self.centralwidget)
        self.rx_elevation_label.setGeometry(QtCore.QRect(20, 730, 121, 17))
        self.rx_elevation_label.setObjectName("rx_elevation_label")
        self.rx_height_label = QtWidgets.QLabel(self.centralwidget)
        self.rx_height_label.setGeometry(QtCore.QRect(20, 700, 121, 17))
        self.rx_height_label.setObjectName("rx_height_label")
        self.atmos_n_scrn_label = QtWidgets.QLabel(self.centralwidget)
        self.atmos_n_scrn_label.setGeometry(QtCore.QRect(20, 510, 181, 17))
        self.atmos_n_scrn_label.setObjectName("atmos_n_scrn_label")
        self.atmos_wind_speed_label = QtWidgets.QLabel(self.centralwidget)
        self.atmos_wind_speed_label.setGeometry(QtCore.QRect(20, 570, 151, 17))
        self.atmos_wind_speed_label.setObjectName("atmos_wind_speed_label")
        self.atmos_fried_r0_label = QtWidgets.QLabel(self.centralwidget)
        self.atmos_fried_r0_label.setGeometry(QtCore.QRect(20, 540, 121, 17))
        self.atmos_fried_r0_label.setObjectName("atmos_fried_r0_label")
        self.atmos_wind_dir_label = QtWidgets.QLabel(self.centralwidget)
        self.atmos_wind_dir_label.setGeometry(QtCore.QRect(20, 600, 161, 17))
        self.atmos_wind_dir_label.setObjectName("atmos_wind_dir_label")
        self.atmos_screen_size_slider = QtWidgets.QSlider(self.centralwidget)
        self.atmos_screen_size_slider.setGeometry(QtCore.QRect(210, 460, 301, 51))
        self.atmos_screen_size_slider.setOrientation(QtCore.Qt.Horizontal)
        self.atmos_screen_size_slider.setObjectName("atmos_screen_size_slider")
        self.atmos_screen_size_label = QtWidgets.QLabel(self.centralwidget)
        self.atmos_screen_size_label.setGeometry(QtCore.QRect(20, 480, 191, 17))
        self.atmos_screen_size_label.setObjectName("atmos_screen_size_label")
        self.sim_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.sim_groupBox.setGeometry(QtCore.QRect(10, 100, 511, 161))
        self.sim_groupBox.setFlat(False)
        self.sim_groupBox.setCheckable(False)
        self.sim_groupBox.setObjectName("sim_groupBox")
        self.sim_grid_scale_input = QtWidgets.QLineEdit(self.sim_groupBox)
        self.sim_grid_scale_input.setGeometry(QtCore.QRect(200, 70, 301, 25))
        self.sim_grid_scale_input.setObjectName("sim_grid_scale_input")
        self.sim_num_inter_input = QtWidgets.QLineEdit(self.sim_groupBox)
        self.sim_num_inter_input.setGeometry(QtCore.QRect(200, 100, 301, 25))
        self.sim_num_inter_input.setObjectName("sim_num_inter_input")
        self.sim_sample_rate_input = QtWidgets.QLineEdit(self.sim_groupBox)
        self.sim_sample_rate_input.setGeometry(QtCore.QRect(200, 130, 301, 25))
        self.sim_sample_rate_input.setObjectName("sim_sample_rate_input")
        self.beam_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.beam_groupBox.setGeometry(QtCore.QRect(10, 280, 511, 151))
        self.beam_groupBox.setObjectName("beam_groupBox")
        self.atmos_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.atmos_groupBox.setGeometry(QtCore.QRect(10, 450, 511, 171))
        self.atmos_groupBox.setObjectName("atmos_groupBox")
        self.atmos_fried_r0_input = QtWidgets.QLineEdit(self.atmos_groupBox)
        self.atmos_fried_r0_input.setGeometry(QtCore.QRect(200, 80, 301, 25))
        self.atmos_fried_r0_input.setObjectName("atmos_fried_r0_input")
        self.atmos_wind_speed_input = QtWidgets.QLineEdit(self.atmos_groupBox)
        self.atmos_wind_speed_input.setGeometry(QtCore.QRect(200, 110, 301, 25))
        self.atmos_wind_speed_input.setObjectName("atmos_wind_speed_input")
        self.atmos_wind_dir_input = QtWidgets.QLineEdit(self.atmos_groupBox)
        self.atmos_wind_dir_input.setGeometry(QtCore.QRect(200, 140, 301, 25))
        self.atmos_wind_dir_input.setObjectName("atmos_wind_dir_input")
        self.atmos_n_scrn_input = QtWidgets.QLineEdit(self.atmos_groupBox)
        self.atmos_n_scrn_input.setGeometry(QtCore.QRect(200, 50, 301, 25))
        self.atmos_n_scrn_input.setObjectName("atmos_n_scrn_input")
        self.rx_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.rx_groupBox.setGeometry(QtCore.QRect(10, 640, 511, 121))
        self.rx_groupBox.setObjectName("rx_groupBox")
        self.run_sim_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.run_sim_groupBox.setGeometry(QtCore.QRect(10, 780, 511, 61))
        self.run_sim_groupBox.setObjectName("run_sim_groupBox")
        self.start_sim_pushButton = QtWidgets.QPushButton(self.run_sim_groupBox)
        self.start_sim_pushButton.setGeometry(QtCore.QRect(110, 30, 89, 25))
        self.start_sim_pushButton.setObjectName("start_sim_pushButton")
        self.restart_sim_pushButton = QtWidgets.QPushButton(self.run_sim_groupBox)
        self.restart_sim_pushButton.setGeometry(QtCore.QRect(310, 30, 89, 25))
        self.restart_sim_pushButton.setObjectName("restart_sim_pushButton")
        self.save_sim_pushButton = QtWidgets.QPushButton(self.run_sim_groupBox)
        self.save_sim_pushButton.setGeometry(QtCore.QRect(410, 30, 89, 25))
        self.save_sim_pushButton.setObjectName("save_sim_pushButton")
        self.init_sim_pushButton = QtWidgets.QPushButton(self.run_sim_groupBox)
        self.init_sim_pushButton.setGeometry(QtCore.QRect(10, 30, 89, 25))
        self.init_sim_pushButton.setObjectName("init_sim_pushButton")
        self.stop_sim_pushButton = QtWidgets.QPushButton(self.run_sim_groupBox)
        self.stop_sim_pushButton.setGeometry(QtCore.QRect(210, 30, 89, 25))
        self.stop_sim_pushButton.setObjectName("stop_sim_pushButton")
        self.rx_int_dist_frame = QtWidgets.QFrame(self.centralwidget)
        self.rx_int_dist_frame.setGeometry(QtCore.QRect(550, 40, 561, 261))
        self.rx_int_dist_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.rx_int_dist_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.rx_int_dist_frame.setObjectName("rx_int_dist_frame")
        self.phase_screen_frame = QtWidgets.QFrame(self.centralwidget)
        self.phase_screen_frame.setGeometry(QtCore.QRect(550, 340, 561, 261))
        self.phase_screen_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.phase_screen_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.phase_screen_frame.setObjectName("phase_screen_frame")
        self.rx_int_distr_label = QtWidgets.QLabel(self.centralwidget)
        self.rx_int_distr_label.setGeometry(QtCore.QRect(550, 20, 231, 17))
        self.rx_int_distr_label.setObjectName("rx_int_distr_label")
        self.phase_screen_label = QtWidgets.QLabel(self.centralwidget)
        self.phase_screen_label.setGeometry(QtCore.QRect(550, 320, 231, 17))
        self.phase_screen_label.setObjectName("phase_screen_label")
        self.perform_metrics_label = QtWidgets.QLabel(self.centralwidget)
        self.perform_metrics_label.setGeometry(QtCore.QRect(550, 610, 231, 17))
        self.perform_metrics_label.setObjectName("perform_metrics_label")
        self.perform_metrics_frame = QtWidgets.QFrame(self.centralwidget)
        self.perform_metrics_frame.setGeometry(QtCore.QRect(550, 630, 561, 211))
        self.perform_metrics_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.perform_metrics_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.perform_metrics_frame.setObjectName("perform_metrics_frame")
        self.verticalScrollBar = QtWidgets.QScrollBar(self.perform_metrics_frame)
        self.verticalScrollBar.setGeometry(QtCore.QRect(540, 0, 21, 211))
        self.verticalScrollBar.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")
        self.rx_ap_diam_input = QtWidgets.QLineEdit(self.centralwidget)
        self.rx_ap_diam_input.setGeometry(QtCore.QRect(210, 670, 301, 25))
        self.rx_ap_diam_input.setObjectName("rx_ap_diam_input")
        self.rx_height_input = QtWidgets.QLineEdit(self.centralwidget)
        self.rx_height_input.setGeometry(QtCore.QRect(210, 700, 301, 25))
        self.rx_height_input.setObjectName("rx_height_input")
        self.rx_elevation_input = QtWidgets.QLineEdit(self.centralwidget)
        self.rx_elevation_input.setGeometry(QtCore.QRect(210, 730, 301, 25))
        self.rx_elevation_input.setObjectName("rx_elevation_input")
        self.beam_power_input = QtWidgets.QLineEdit(self.centralwidget)
        self.beam_power_input.setGeometry(QtCore.QRect(210, 310, 301, 25))
        self.beam_power_input.setObjectName("beam_power_input")
        self.beam_wvl_input = QtWidgets.QLineEdit(self.centralwidget)
        self.beam_wvl_input.setGeometry(QtCore.QRect(210, 340, 301, 25))
        self.beam_wvl_input.setObjectName("beam_wvl_input")
        self.beam_waist_input = QtWidgets.QLineEdit(self.centralwidget)
        self.beam_waist_input.setGeometry(QtCore.QRect(210, 370, 301, 25))
        self.beam_waist_input.setObjectName("beam_waist_input")
        self.sim_prog_bar = QtWidgets.QProgressBar(self.centralwidget)
        self.sim_prog_bar.setGeometry(QtCore.QRect(20, 60, 491, 21))
        self.sim_prog_bar.setProperty("value", 24)
        self.sim_prog_bar.setObjectName("sim_prog_bar")
        self.sim_prog_label = QtWidgets.QLabel(self.centralwidget)
        self.sim_prog_label.setGeometry(QtCore.QRect(20, 30, 491, 21))
        self.sim_prog_label.setText("")
        self.sim_prog_label.setObjectName("sim_prog_label")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(9, -1, 511, 91))
        self.groupBox.setObjectName("groupBox")
        self.groupBox.raise_()
        self.run_sim_groupBox.raise_()
        self.rx_groupBox.raise_()
        self.atmos_groupBox.raise_()
        self.beam_groupBox.raise_()
        self.sim_groupBox.raise_()
        self.beam_power_label.raise_()
        self.beam_wvl_label.raise_()
        self.beam_waist_label.raise_()
        self.beam_prop_dir_label.raise_()
        self.beam_prop_dir_box.raise_()
        self.sim_grid_size_slider.raise_()
        self.sim_grid_size_label.raise_()
        self.sim_num_inter_label.raise_()
        self.sim_grid_scale_label.raise_()
        self.sim_sample_rate_label.raise_()
        self.rx_ap_diam_label.raise_()
        self.rx_elevation_label.raise_()
        self.rx_height_label.raise_()
        self.atmos_n_scrn_label.raise_()
        self.atmos_wind_speed_label.raise_()
        self.atmos_fried_r0_label.raise_()
        self.atmos_wind_dir_label.raise_()
        self.atmos_screen_size_slider.raise_()
        self.atmos_screen_size_label.raise_()
        self.rx_int_dist_frame.raise_()
        self.phase_screen_frame.raise_()
        self.rx_int_distr_label.raise_()
        self.phase_screen_label.raise_()
        self.perform_metrics_label.raise_()
        self.perform_metrics_frame.raise_()
        self.rx_ap_diam_input.raise_()
        self.rx_height_input.raise_()
        self.rx_elevation_input.raise_()
        self.beam_power_input.raise_()
        self.beam_wvl_input.raise_()
        self.beam_waist_input.raise_()
        self.sim_prog_bar.raise_()
        self.sim_prog_label.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1122, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FSO simulation tool"))
        self.beam_power_label.setText(_translate("MainWindow", "Power  (mW)"))
        self.beam_wvl_label.setText(_translate("MainWindow", "Wavelength  (nm)"))
        self.beam_waist_label.setText(_translate("MainWindow", "Beam waist (mm)"))
        self.beam_prop_dir_label.setText(_translate("MainWindow", "Proagation direction"))
        self.beam_prop_dir_box.setItemText(0, _translate("MainWindow", "Uplink"))
        self.beam_prop_dir_box.setItemText(1, _translate("MainWindow", "Downlink"))
        self.sim_grid_size_label.setText(_translate("MainWindow", "Grid size 512 (pxl x pxl)"))
        self.sim_num_inter_label.setText(_translate("MainWindow", "Number of iterations "))
        self.sim_grid_scale_label.setText(_translate("MainWindow", "Grid scale (m x m)"))
        self.sim_sample_rate_label.setText(_translate("MainWindow", "Sampling rate (s)"))
        self.rx_ap_diam_label.setText(_translate("MainWindow", "Aperture diam.  (mm)"))
        self.rx_elevation_label.setText(_translate("MainWindow", "Elevation  (deg)"))
        self.rx_height_label.setText(_translate("MainWindow", "Height (km)"))
        self.atmos_n_scrn_label.setText(_translate("MainWindow", "Number of screens "))
        self.atmos_wind_speed_label.setText(_translate("MainWindow", "Wind speed (m/s)"))
        self.atmos_fried_r0_label.setText(_translate("MainWindow", "Fried r0  (cm)"))
        self.atmos_wind_dir_label.setText(_translate("MainWindow", "Wind direction  (deg)"))
        self.atmos_screen_size_label.setText(_translate("MainWindow", "Screen size 1024 (pxl x pxl)"))
        self.sim_groupBox.setTitle(_translate("MainWindow", "SIMULATION"))
        self.beam_groupBox.setTitle(_translate("MainWindow", "OPTICAL BEAM"))
        self.atmos_groupBox.setTitle(_translate("MainWindow", "ATMOSPHERE"))
        self.rx_groupBox.setTitle(_translate("MainWindow", "RECEIVER"))
        self.run_sim_groupBox.setTitle(_translate("MainWindow", "SIMULATION CONTROL"))
        self.start_sim_pushButton.setText(_translate("MainWindow", "START"))
        self.restart_sim_pushButton.setText(_translate("MainWindow", "RESTART"))
        self.save_sim_pushButton.setText(_translate("MainWindow", "SAVE"))
        self.init_sim_pushButton.setText(_translate("MainWindow", "INIT"))
        self.stop_sim_pushButton.setText(_translate("MainWindow", "STOP"))
        self.rx_int_distr_label.setText(_translate("MainWindow", "Receiver Intensity distribution"))
        self.phase_screen_label.setText(_translate("MainWindow", "Phase screens"))
        self.perform_metrics_label.setText(_translate("MainWindow", "Performance metrics"))
        self.groupBox.setTitle(_translate("MainWindow", "SIMULATION PROGRESS"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())