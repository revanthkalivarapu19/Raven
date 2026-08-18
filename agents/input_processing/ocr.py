import easyocr
import cv2

reader = easyocr.Reader(['en', 'te'])

def preprocess_image(image_path: str):
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def extract_text_from_image(image_path: str) -> str:
    preprocessed = preprocess_image(image_path)
    results = reader.readtext(preprocessed)
    extracted_text = " ".join([text for (_, text, _) in results])
    return extracted_text

if __name__ == "__main__":
    text = extract_text_from_image("testimage_tel.png")
    print("Extracted text:", text)
