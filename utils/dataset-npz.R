# TODO: Add tfruns functionality 
# TODO: Add callbacks for lr_plateau, save weights, etc
# TODO: lr range test
# TODO: Fix dataset pipeline so we don't have to load entire data in-memory
# TODO: Make model smaller

source_dir <- "/home/jason/kaggle/bengali/data/data-npz/resized_64"

np <- reticulate::import("numpy")

train0 <- np$load(file.path(source_dir, "train_images0.npz"))
train1 <- np$load(file.path(source_dir, "train_images1.npz"))
# train2 <- np$load(file.path(source_dir, "train_images2.npz"))
# train3 <- np$load(file.path(source_dir, "train_images3.npz"))

root0  <- np$load(file.path(source_dir, "train_root_labels0.npz"))$f['train_root_labels']
cons0  <- np$load(file.path(source_dir, "train_consonant_labels0.npz"))$f['train_consonant_labels']
vowel0 <- np$load(file.path(source_dir, "train_vowel_labels0.npz"))$f['train_vowel_labels']

root1  <- np$load(file.path(source_dir, "train_root_labels1.npz"))$f['train_root_labels']
cons1  <- np$load(file.path(source_dir, "train_consonant_labels1.npz"))$f['train_consonant_labels']
vowel1 <- np$load(file.path(source_dir, "train_vowel_labels1.npz"))$f['train_vowel_labels']

# root2  <- np$load(file.path(source_dir, "train_root_labels2.npz"))$f['train_root_labels']
# cons2  <- np$load(file.path(source_dir, "train_consonant_labels2.npz"))$f['train_consonant_labels']
# vowel2 <- np$load(file.path(source_dir, "train_vowel_labels2.npz"))$f['train_vowel_labels']
# 
# root3  <- np$load(file.path(source_dir, "train_root_labels3.npz"))$f['train_root_labels']
# cons3  <- np$load(file.path(source_dir, "train_consonant_labels3.npz"))$f['train_consonant_labels']
# vowel3 <- np$load(file.path(source_dir, "train_vowel_labels3.npz"))$f['train_vowel_labels']


# y <- train2$f['train_images']
# z <- train3$f['train_images']

npz0 <- train0$f['train_images']
npz1 <- train1$f['train_images']

# x <- as_tensor(npz0)
# y <- as_tensor(npz1)  

roots  <- tf$concat(list(root0, root1),   axis = 0L)
vowels <- tf$concat(list(vowel0, vowel1), axis = 0L)
cons   <- tf$concat(list(cons0, cons1),   axis = 0L)

x_train <- tf$concat(list(npz0, npz1), axis = 0L)



ds <- tf$data$Dataset$from_tensor_slices(tuple(x_train, tuple(roots, cons, vowels))) %>% 
  dataset_batch(FLAGS$batch_size, drop_remainder = TRUE) %>% 
  dataset_prefetch(10L)

val_ds <- ds$take(100L)

ds %<>% dataset_repeat()

rm(x_train)
rm(npz0)
rm(npz1)
# Potentially serialze from here (tfrecor) to avoid fitting whole ds in-memory?