"""Script for loading EasyOCR model.
Build on EasyOCR library accessible there : https://github.com/JaidedAI/EasyOCR
"""

import easyocr
import numpy as np 

class EasyOCRModel :
    def __init__(self, device) :
        self.model = easyocr.Reader(['en'], gpu=True if device == 'cuda' else False)

    def compute_ocr(
            self,
            generated_images
    ) :
        generated_words = []

        for image in generated_images :
            result = self.model.readtext(np.array(image), detail=0, decoder="beamsearch")

            word = " ".join(result).lower().strip()
            generated_words.append(word)

        return generated_words