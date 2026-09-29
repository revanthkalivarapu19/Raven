from agents.input_processing.ocr import extract_text_from_image


def main():

    image_path = "agents/input_processing/testimage.png"

    extracted_text = extract_text_from_image(image_path)

    print("\n========== OCR RESULT ==========")
    print("Extracted Text:")
    print(extracted_text)


if __name__ == "__main__":
    main()