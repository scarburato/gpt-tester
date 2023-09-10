import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import requests
import mimetypes

from main_window import Ui_MainWindow

def run_file(filePath, restURL):
  try:
    # Open the file for reading
    with open(filePath, 'r') as file:
        file_content = file.read()

    # Create a dictionary containing the data to send in the request
    payload = {
      'text': file_content
    }

    # Send a POST request to the REST endpoint
    response = requests.post(restURL, json=payload)

    # Check the response status code
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Failed to send file content. Status code: {response.status_code}')
  except FileNotFoundError:
    print(f'File not found: {filePath}')
  except Exception as e:
    print(f'An error occurred: {str(e)}')

def num_file(dir):
  if dir is None:
      return 0
  count = 0
  for file_name in os.listdir(dir):
    file_path = os.path.join(dir, file_name)
    if os.path.isdir(file_path):
      count += num_file(file_path)
    if not os.path.isfile(file_path):# or not mimetypes.guess_type(file_path)[0] == 'text/plain':
      continue
    count += 1
  return count


class BonGTPFileSystemModel(QFileSystemModel):

  def columnCount(self, parent=None, check=lambda file: "???", *args, **kwargs):
    self.check = check
    self.myIndex = super(BonGTPFileSystemModel, self).columnCount()
    return self.myIndex + 1

  def headerData(self, section, orientation, role=None):
    if section != self.myIndex or orientation != Qt.Orientation.Horizontal or role != Qt.ItemDataRole.DisplayRole:
      return super(BonGTPFileSystemModel, self).headerData(section, orientation, role)
    return "isGPT"

  def data(self, index, role=None):
    if index.column() == self.myIndex:
      if role == Qt.ItemDataRole.DisplayRole:
        path = index.data(QFileSystemModel.FilePathRole)
        return self.check(path)

      if role == Qt.ItemDataRole.TextAlignmentRole:
        return Qt.AlignmentFlag.AlignLeft

    return super(BonGTPFileSystemModel, self).data(index, role)


class BonGTPWindow():
  def __init__(self):
    # Props
    self.currentDir = None
    self.results = {}

    # Set-up window
    self.mw = QMainWindow(None)
    self.uimw = Ui_MainWindow()
    self.uimw.setupUi(self.mw)
    self.uimw.progressBar.setValue(0)

    # Triggers
    self.uimw.actionOpen_Folder.triggered.connect(lambda: self.openFolderTrigger())
    self.uimw.pushButtonOpenFolder.clicked.connect(lambda: self.openFolderTrigger())
    self.uimw.pushButtonRun.clicked.connect(lambda: self.run_all_dir(self.currentDir, "http://localhost:5000/evaluate"))

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

    self.dirModel = BonGTPFileSystemModel(None)
    self.dirModel.setRootPath(QDir.rootPath())
    self.dirModel.setFilter(QDir.NoDotAndDotDot | QDir.Filter.AllEntries)

    root_index = self.dirModel.index(dir)

    self.proxy = QSortFilterProxyModel(self.dirModel)
    self.proxy.setSourceModel(self.dirModel)

    self.uimw.treeView.setModel(self.proxy)

    proxy_root_index = self.proxy.mapFromSource(root_index)
    self.uimw.treeView.setRootIndex(proxy_root_index)

    # Hide all columns except the first, the name column
    for i in range(1, self.dirModel.columnCount() - 1):
      self.uimw.treeView.hideColumn(i)

    self.uimw.treeView.setHeaderHidden(False)
    self.uimw.treeView.clicked.connect(lambda e: self.update_text_preview(e))

  def update_text_preview(self, fileIndex: QModelIndex):
    ix = self.proxy.mapToSource(fileIndex)
    self.currentFile = ix.data(QFileSystemModel.FilePathRole)

    if(not os.path.isfile(self.currentFile)):
      return

    with open(self.currentFile) as file:
      self.uimw.textPreview.setPlainText(file.read())

    if self.currentFile in self.results:
      self.uimw.textProcessStatus.setText(str(self.results[self.currentFile]))
    else:
      self.uimw.textProcessStatus.setText("Ready")

  def update_index_column(self, filepath):
    if not os.path.isfile(filepath):
      return ""

    if filepath not in self.results:
      return "???"

    return self.results[filepath]

  def clear_all(self):
    self.uimw.textPreview.clear()
    self.currentFile = None
    self.currentDir = None
    self.results = {}

  def run_all_dir(self, dir, restURL):
    if dir is None:
      return

    self.uimw.progressBar.setValue(0)
    self.uimw.progressBar.setMaximum(num_file(dir))


    for file_name in os.listdir(dir):
      file_path = os.path.join(dir, file_name)

      # Recursive step
      if os.path.isdir(file_path):
        self.run_all_dir(file_path, restURL)
        continue

      if not os.path.isfile(file_path):# or not mimetypes.guess_type(file_path)[0] == 'text/plain':
        continue

      result = run_file(file_path, restURL)
      self.results[file_path] = result

      print(result)

      # Progress
      self.uimw.progressBar.setValue(self.uimw.progressBar.value() + 1)
      self.dirModel.setRootPath(self.dirModel.rootPath())
      self.uimw.treeView.update()

if __name__ == '__main__':
  app = QApplication(sys.argv)

  bonGTP = BonGTPWindow()
  bonGTP.show()


  app.exec()
  pass