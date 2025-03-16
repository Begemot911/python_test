import cv2

# Загрузка предварительно обученных моделей для распознавания лиц и ключевых точек
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('lbfmodel.yaml')

# Инициализация видеозахвата
cap = cv2.VideoCapture(0)

while True:
    # Захват кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Для каждого обнаруженного лица
    for (x, y, w, h) in faces:
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Обнаружение ключевых точек лица
        ok, landmarks = facemark.fit(gray, faces=[[x, y, w, h]])

        if ok:
            # Рисуем ключевые точки
            for landmark in landmarks:
                for (x, y) in landmark[0]:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

    # Отображение кадра
    cv2.imshow('Face Detection', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()