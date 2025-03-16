import cv2

# Загрузка детектора лиц и модели LBF
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")  # Укажите путь к файлу lbfmodel.yaml

# Захват видео с камеры
cap = cv2.VideoCapture(0)

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

        # Получение ключевых точек
        ok, landmarks = facemark.fit(gray, faces=[[x, y, w, h]])

        if ok:
            # Рисуем ключевые точки
            for landmark in landmarks:
                for (x, y) in landmark[0]:
                    cv2.circle(resized_frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Отображение кадра
    cv2.imshow('Face Landmarks', resized_frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
