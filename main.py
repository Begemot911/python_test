import cv2
from flask import Flask, Response

# Инициализация Flask
app = Flask(__name__)

# Загрузка каскада Хаара для распознавания лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    # Инициализация видеозахвата
    cap = cv2.VideoCapture(0)

    while True:
        # Захват кадра
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование кадра в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Распознавание лиц
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Рисуем прямоугольники вокруг лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Преобразуем кадр в JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Возвращаем кадр в виде байтов
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Освобождение ресурсов
    cap.release()

# Маршрут для вывода видео
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Запуск сервера
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
