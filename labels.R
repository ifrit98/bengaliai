


class_map <- readr::read_csv(
  "data-raw/class_map.csv",
  skip = 0L,
  col_types = list(col_character(),
                   col_integer(),
                   col_character())
)

train_lab <- readr::read_csv("data-raw/train.csv")
test_lab  <- readr::read_csv("data-raw/test.csv")
sub       <- readr::read_csv("data-raw/sample_submission.csv")

names(class_map) <- c("type", "index", "root")

cm <- class_map
tl <- train_lab
ts <- test_lab

grapheme_idx <- 1:167
vowel_diacritic_idx <- 169:179
consonant_diacritic_idx <- 180:186




GRAPHEMES  <- class_map[grapheme_idx,]
VOWELS     <- class_map[vowel_diacritic_idx,]
CONSONANTS <- class_map[consonant_diacritic_idx,]


G <- tf$constant(GRAPHEMES$index, dtype = tf$int32)
V <- tf$constant(VOWELS$index, dtype = tf$int32)
C <- tf$constant(CONSONANTS$index, dtype = tf$int32)



# library(forcats)
# 
# factor(tl$grapheme, levels = cm$root)
# as.factor(tl$grapheme)
# 
# VOWELS
# a <- tl[1,]
# GRAPHEMES[1,]
# 
# tl$grapheme
