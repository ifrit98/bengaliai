

grapheme_labels <- 0:167
vowel_diacritic_labels <- 0:10
consonant_diacritic_labels <- 0:6

class_map <- readr::read_csv("data-raw/class_map.csv", skip = 1L)

train_labels <- readr::read_csv("data-raw/train.csv")
test_labels  <- readr::read_csv("data-raw/test.csv")

submission <- readr::read_csv("data-raw/sample_submission.csv")
