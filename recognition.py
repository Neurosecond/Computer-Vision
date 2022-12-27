import pytesseract
import numpy as np
import cv2

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
tessdata_dir_config = '--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata"'


def recognition():
    image = cv2.imread('resources/aventa_crop2.jpg')
    # Получаем контрастное изображение в серых тонах
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image = cv2.inRange(image, (10., 85., 130.), (179., 255., 255.))
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Устраняем шумы
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    # Конвертируем в бинарный формат с использованием automatic threshold (use cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Расширяем thresh для объединения текстовых областей в блоки строк.
    dilated_thresh = cv2.dilate(thresh, np.ones((3, 100)), iterations=1)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # thresh = cv2.resize(thresh, (1920 // 2, 1080 // 2))
    # opening = cv2.resize(opening, (1920 // 2, 1080 // 2))
    # invert = 255 - opening

    # Находим контуры dilated_thresh
    cnts = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # Use index [-2] to be compatible to OpenCV 3 and 4

    # Создаем список ограничивающих рамок
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]

    # Сортируем ограничивающие рамки как "top to bottom" по индексу 1
    bounding_boxes = sorted(bounding_boxes, key=lambda bb: bb[1])

    # Итерируем ограничивающие рамки
    for b in bounding_boxes:
        x, y, w, h = b

        if (h > 30) and (w > 70):
            # Обрезаем slice, и инвертируем black and white (tesseract prefers black text).
            thresh_croped = thresh[max(y - 10, 0):min(y + h + 10, thresh.shape[0]),
                            max(x - 10, 0):min(x + w + 10, thresh.shape[1])]
            slice = 255 - thresh_croped
            text = pytesseract \
                .image_to_boxes(slice,
                                config="-c tessedit"
                                       "_char_whitelist"
                                       "=1234567890-:./ACV "
                                       " --psm 3"
                                       " --tessdata-dir 'C:/Program Files (x86)/Tesseract-OCR/tessdata'")

            print(text)
            # Перебираем данные про текстовые надписи
            for i, el in enumerate(text.splitlines()):
                if i == 0:
                    continue

                el = el.split()
                try:
                    # Создаем подписи на картинке
                    x, y, w, h = int(el[1]), int(el[2]), int(el[3]), int(el[4])
                    cv2.rectangle(slice, (x, y), (w + x, h + y), (0, 0, 255), 1)
                    cv2.putText(slice, el[0], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
                except IndexError:
                    print("Операция была пропущена")

            cv2.imshow('slice', slice)
            cv2.waitKey()

    cv2.imshow('img', thresh)
    # cv2.imwrite('resources/aventa_crop.jpg', invert)
    cv2.waitKey()


if __name__ == "__main__":
    recognition()
