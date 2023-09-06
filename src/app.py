import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from main_window import Ui_MainWindow

class BonGTPWindow():
  def __init__(self):
    # Props
    self.currentDir = None

    # Set-up window
    self.mw = QMainWindow(None)
    self.uimw = Ui_MainWindow()
    self.uimw.setupUi(self.mw)

    # Triggers
    self.uimw.actionOpen_Folder.triggered.connect(lambda: self.openFolderTrigger())
    self.uimw.pushButtonOpenFolder.clicked.connect(lambda: self.openFolderTrigger())

  def show(self):
    self.mw.show()

  def openFolderTrigger(self):
    self.clear_all()

    # Folder dialog
    dir = QFileDialog.getExistingDirectory(
      self.mw,
      "Open a folder",
      "../..", # os.path.expanduser("~"),
    )
    self.currentDir = dir if len(dir) > 0 else None
    print(self.currentDir)

    self.dirModel = QFileSystemModel(None)
    self.dirModel.setRootPath(QDir.rootPath())
    self.dirModel.setFilter(QDir.NoDotAndDotDot | QDir.Filter.AllEntries)

    root_index = self.dirModel.index(dir)

    self.proxy = QSortFilterProxyModel(self.dirModel)
    self.proxy.setSourceModel(self.dirModel)

    self.uimw.treeView.setModel(self.proxy)

    proxy_root_index = self.proxy.mapFromSource(root_index)
    self.uimw.treeView.setRootIndex(proxy_root_index)

    self.uimw.treeView.setHeaderHidden(True)
    self.uimw.treeView.clicked.connect(lambda e: self.update_text_preview(e))

  def update_text_preview(self, fileIndex: QModelIndex):
    ix = self.proxy.mapToSource(fileIndex)
    self.currentFile = ix.data(QFileSystemModel.FilePathRole)

    if(not os.path.isfile(self.currentFile)):
      return

    with open(self.currentFile) as file:
      self.uimw.textPreview.setPlainText(file.read())

  def clear_all(self):
    self.uimw.textPreview.clear()
    self.currentFile = None
    self.currentDir = None

if __name__ == '__main__':
  app = QApplication(sys.argv)

  bonGTP = BonGTPWindow()
  bonGTP.show()


  app.exec()
  pass