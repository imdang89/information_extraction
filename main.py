from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSerialPort import *
from PyQt5.QtCore import *
from cv2 import cvtColor, waitKey
from camera import Camera
from gui import *
import cv2
import sys
import numpy as np
import uuid
from extrac_result import *
import csv
import time


cam = Camera(2)
cam.initialize()

class runCamera(QThread):
    ImgUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            self.frame = cam.get_frame()
            #ĐỔI MÀU TỪ (b, g, r) -> (r, g, b)
            self.Image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.ImgQt = QImage(self.Image.data, self.Image.shape[1], self.Image.shape[0], QImage.Format_RGB888)
            # Rezise ảnh về kích thước 640x480
            self.Pict = self.ImgQt.scaled(640, 480, Qt.KeepAspectRatio)
            self.ImgUpdate.emit(self.Pict)
    def stop(self):
        self.ThreadActive = False
        self.quit()


class myUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_SME_HUST()
        self.ui.setupUi(self)


        self.ui.btnConnect.clicked.connect(self.buttonConnect)
        self.ui.btnDisconnect.clicked.connect(self.buttonDisConnect)
        self.ui.btnProcessing.clicked.connect( self.buttonProcessing)
        self.ui.btnExport.clicked.connect(self.buttonExport2excel)

    def buttonConnect(self):
        self.cam = runCamera()
        self.cam.start()
        self.cam.ImgUpdate.connect(self.updateImage)

    def updateImage(self, image):
        self.ui.picLive.setPixmap(QPixmap.fromImage(image))


    def buttonDisConnect(self):
        self.cam.stop()
        self.cam.quit()
        self.cam


    def buttonProcessing(self):
        # xóa dữ liệu thông tin trên list widget
        self.ui.listWidget.clear()

        start_time = time.time()

        self.image = cam.get_frame()
        self.cam.stop()
        cv2.imwrite(f"images\\mcocr_private_145120aorof.jpg", self.image)

        img_path = 'images\\mcocr_private_145120aorof.jpg'
        random_id = str(uuid.uuid4())
        image = np.array(Image.open(img_path))
        img_info = inf(image, random_id)
        save_dir = cf.result_img_dir
        img, vis_img,self.kie_info = infer(img_info, save_dir, random_id)
        # resize ảnh
        ratio = np.minimum(407/vis_img.shape[0], 512/vis_img.shape[1])
        vis_img = cv2.resize(vis_img, dsize= None, fx=ratio, fy= ratio)

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.ui.lblSpeed.clear()
        self.ui.lblSpeed.setText("Process Speed: "+ str(np.round(elapsed_time,2 )) +"(s)")

        cv2.imwrite(f"checked_image\\vis_img_resize.jpg", vis_img)
        # hiển thị ảnh sau khi resize và căn giữa ảnh
        self.ui.picResult.setPixmap(QPixmap(f"checked_image\\vis_img_resize.jpg"))
        
        print("elapsed_time",elapsed_time)
        
        for key, value in self.kie_info.items():
            if key == 'Dia Chi':
                key = "Địa chỉ"
            elif key == "Noi Ban":
                key = "Nơi bán"
            elif key =="Thoi Gian":
                key == "Thời gian"
            elif key =="Tong So Tien":
                key == "Tổng số tiền"
            item = QListWidgetItem(key + ": " + value)
            self.ui.listWidget.addItem(item)

    def buttonExport2excel(self):
        with open('data.csv', mode='w', encoding='utf-8', newline='') as csv_file:
            # Tạo writer object để ghi dữ liệu vào file CSV
            writer = csv.writer(csv_file)
            
            # Ghi dòng tiêu đề vào file CSV
            writer.writerow(self.kie_info.keys())
            
            # Ghi dữ liệu vào file CSV
            for row in zip(*self.kie_info.values()):
                writer.writerow(row)




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = myUI()
    MainWindow.show()
    sys.exit(app.exec_())


