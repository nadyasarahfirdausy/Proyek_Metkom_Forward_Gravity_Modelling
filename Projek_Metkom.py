import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QLineEdit, QLabel, QHBoxLayout, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Konstanta gravitasi universal
G = 6.674e-11  

class GravityForwardGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projek Metkom - Forward Modelling Gravitasi (Dengan Input)")
        self.setGeometry(100, 100, 700, 800)

        # --- Elemen GUI ---
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvas(self.figure)

        # Input fields
        self.input_R = QLineEdit()
        self.input_drho = QLineEdit()
        self.input_z0 = QLineEdit()

        # Placeholder text (petunjuk)
        self.input_R.setPlaceholderText("Masukkan radius bola (m), misal 100")
        self.input_drho.setPlaceholderText("Masukkan kontras densitas Δρ (kg/m³), misal 400")
        self.input_z0.setPlaceholderText("Masukkan kedalaman pusat bola (m), misal 300")

        # Label
        lbl_R = QLabel("Radius (m):")
        lbl_drho = QLabel("Δρ (kg/m³):")
        lbl_z0 = QLabel("Kedalaman (m):")

        # Tombol
        self.btn_compute = QPushButton("Hitung Anomali Gravitasi")
        self.btn_compute.clicked.connect(self.compute_anomaly)

        # Layout input
        input_layout = QVBoxLayout()
        for lbl, inp in [(lbl_R, self.input_R), (lbl_drho, self.input_drho), (lbl_z0, self.input_z0)]:
            hbox = QHBoxLayout()
            hbox.addWidget(lbl)
            hbox.addWidget(inp)
            input_layout.addLayout(hbox)

        # Layout utama
        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.btn_compute)
        main_layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def compute_anomaly(self):
        """Hitung anomali gravitasi dari input pengguna"""
        try:
            R = float(self.input_R.text())
            drho = float(self.input_drho.text())
            z0 = float(self.input_z0.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Pastikan semua input berupa angka!")
            return

        # Grid pengamatan di permukaan
        x = np.linspace(-1000, 1000, 100)
        y = np.linspace(-1000, 1000, 100)
        X, Y = np.meshgrid(x, y)
        Z = z0

        # Rumus anomali gravitasi bola homogen
        gz = (4/3)*np.pi*G*drho*R**3 * Z / ((X**2 + Y**2 + Z**2)**1.5)
        gz_mgal = gz * 1e5  # konversi ke mGal

        # Plot peta anomali gravitasi
        self.ax1.clear()
        im = self.ax1.imshow(gz_mgal, extent=[x.min(), x.max(), y.min(), y.max()],
                             origin='lower', cmap='jet')
        self.ax1.set_title("Peta Anomali Gravitasi (mGal)")
        self.ax1.set_xlabel("X (m)")
        self.ax1.set_ylabel("Y (m)")
        self.figure.colorbar(im, ax=self.ax1, orientation='vertical')

        # Plot profil 1D (potongan tengah)
        self.ax2.clear()
        center_row = gz_mgal[gz_mgal.shape[0] // 2, :]
        self.ax2.plot(x, center_row, color='darkred')
        self.ax2.set_title("Profil Anomali Gravitasi Sepanjang Sumbu X")
        self.ax2.set_xlabel("X (m)")
        self.ax2.set_ylabel("Δg (mGal)")
        self.ax2.grid(True)

        self.canvas.draw()

# Jalankan aplikasi
app = QApplication(sys.argv)
window = GravityForwardGUI()
window.show()
sys.exit(app.exec_())
