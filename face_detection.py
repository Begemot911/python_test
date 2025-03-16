import cv2
from flask import Flask, Response
import numpy as np

app = Flask(__name__)

# Загрузка детектора лиц и модели LBF
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")  # Укажите путь к файлу lbfmodel.yaml

def generate_frames():
    # Захват видео с камеры
    cap = cv2.VideoCapture(1)

    while True:
        # Захват кадра
        ret, frame = cap.read()
        if not ret:
            break

        # Уменьшение разрешения кадра для повышения производительности
        scale_percent = 50  # Уменьшение до 50% от исходного размера
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized_frame = cv2.resize(frame, (width, height))

        # Преобразование в оттенки серого
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Распознавание лиц
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Для каждого обнаруженного лица
        for (x, y, w, h) in faces:
            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_rect=np.array([[x, y, w, h]], dtype=np.int32)

            # Получение ключевых точек
            ok, landmarks = facemark.fit(gray, faces=face_rect)

            if ok:
                # Рисуем ключевые точки
                for landmark in landmarks:
                    for (x, y) in landmark[0]:
                        cv2.circle(resized_frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Преобразуем кадр в JPEG
        ret, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()

        # Возвращаем кадр в виде байтов
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Сервер работает! Перейдите на /video для просмотра видео."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
