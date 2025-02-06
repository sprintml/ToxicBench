"""Main script for loading ocr models."""

import easyocr
from ocr_models import easy_ocr

class OCRModel:
    def __init__(self, model_name, device) :
        self.device = device        
        self.model = self.model_from_name(model_name)

    def model_from_name(self, model_name) :
        if model_name == "easyocr" :
            ocr_model = easy_ocr.EasyOCRModel(self.device)
        else : 
            raise(TypeError("OCR model not supported. See README for supported models."))
        return ocr_model
