import cv2
import numpy as np
from keras.models import load_model

# Tải mô hình Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tải mô hình CNN đã được huấn luyện (định dạng .h5)
model = load_model('path_to_your_cnn_model.h5')

# Hàm phát hiện khuôn mặt và nhận diện
def detect_and_recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Tiền xử lý khuôn mặt cho CNN
        face_resized = cv2.resize(roi_gray, (64, 64))  # Điều chỉnh kích thước theo mô hình của bạn
        face_normalized = face_resized / 255.0
        face_reshaped = np.reshape(face_normalized, (1, 64, 64, 1))  # Điều chỉnh định dạng theo mô hình của bạn

        # Dự đoán nhãn của khuôn mặt
        prediction = model.predict(face_reshaped)
        label = np.argmax(prediction, axis=1)[0]  # Lấy nhãn có xác suất cao nhất

        # Vẽ hình chữ nhật xung quanh khuôn mặt và nhãn
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Khởi động camera và chạy phát hiện khuôn mặt
cap = cv2.VideoCapture(0)  # 0 là camera mặc định

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện và nhận diện khuôn mặt
    frame = detect_and_recognize_faces(frame)

    cv2.imshow('Face Recognition', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
