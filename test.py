import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QApplication

app = QApplication([])
view = gl.GLViewWidget()
view.show()
app.exec_()