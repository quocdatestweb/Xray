# -*- coding: utf-8 -*-
import pickle
from flask import Flask, render_template, request
import os
from random import random
from my_yolov6 import my_yolov6
import cv2

yolov6_model = my_yolov6("weights/best_ckpt.pt","cpu","data/vinbigdata.yaml",640, True)

# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                # Convert image to dest size tensor
                frame = cv2.imread(path_to_save)

                frame, ndet, labels = yolov6_model.infer(frame, conf_thres=0.3, iou_thres=0.4)

                if ndet!=0:
                    cv2.imwrite(path_to_save, frame)

                    # Trả về kết quả
                    return render_template("index.html", user_image = image.filename , rand = str(random()),
                                           msg="Tải file lên thành công", ndet = ndet, label=labels)
                else:
                    return render_template('index.html',user_image = image.filename , rand = str(random()), msg='Không nhận diện được bệnh', ndet = ndet)
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên', ndet = ndet)

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được bệnh')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)

if __name__ == '__main__':
    app.run()
