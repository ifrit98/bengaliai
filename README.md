## Bengali Handwritten Grapheme Classification

Bengali is the 5th most spoken language in the world with hundreds of million of speakers. It’s the official language of Bangladesh and the second most spoken language in India. Considering its reach, there’s significant business and educational interest in developing AI that can optically recognize images of the language handwritten. This challenge hopes to improve on approaches to Bengali recognition.

Optical character recognition is particularly challenging for Bengali. While Bengali has 49 letters (to be more specific 11 vowels and 38 consonants) in its alphabet, there are also 18 potential diacritics, or accents. This means that there are many more graphemes, or the smallest units in a written language. The added complexity results in ~13,000 different grapheme variations (compared to English’s 250 graphemic units).

Dataset comes from Bengali.AI, which hopes to democratize and accelerate research in Bengali language technologies and promote machine learning education.

This dataset contains the image of a handwritten Bengali grapheme and this model separately classifies the three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.


### Data Pipeline
```
 | parquet (.pq) files (32332,)  |
              -> 
 | invert(img) { 255 - img }     |
              -> 
 | Crop & Resize (128, 128)      |
              -> 
 | augment(x, img) { op(img) }   |  # Augment ops: autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, 
              ->                    #              translate_x, translate_y, color, contrast, brightness, sharpness
 | normalize(img) {              |
 |   (img - mean(img)) / sd(img) |
 |  }                            | 
              -> 
 | scale(img) { img / max(img) } |
```
 
### Model Architecture
#### Squeeze and Excitation Networks
![alt_text](HYPERLINK "Diagram of a Squeeze-and-Excitation building block")
- [arXiv 1709.01507](https://arxiv.org/abs/1709.01507)

![alt text](https://raw.githubusercontent.com/ifrit98/bengaliai/master/plots/history.png "Fit Call History ~180 epochs")

