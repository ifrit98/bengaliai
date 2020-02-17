

class_map <- readr::read_csv(
  "csv/class_map.csv",
  skip = 0L,
  col_types = list(col_character(),
                   col_integer(),
                   col_character())
)

train_lab <- readr::read_csv("csv/train.csv")
test_lab  <- readr::read_csv("csv/test.csv")
sub       <- readr::read_csv("csv/sample_submission.csv")

names(class_map) <- c("type", "index", "root")

cm <- class_map
tl <- train_lab
ts <- test_lab

grapheme_idx <- 1:168
vowel_diacritic_idx <- 169:179
consonant_diacritic_idx <- 180:186




GRAPHEMES  <- class_map[grapheme_idx,]
VOWELS     <- class_map[vowel_diacritic_idx,]
CONSONANTS <- class_map[consonant_diacritic_idx,]


GPH <- GRAPHEMES
VOW <- VOWELS
CON <- CONSONANTS

