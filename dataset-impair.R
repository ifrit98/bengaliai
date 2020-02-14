
import_from("dataset.R", ds_raw)

# TODO: VALIDATE images in python before serializing to TFRECORD. 
# TODO: Call python image augmentation functions here or do pre-serialization?
# TODO: Create tf.functions of python functions for dataset.map calls
# TODO: rework TFRECORD pipeline to include augmentation
#         (augmix, normalize, invert, etc)
# TODO: Keep original images as well in dataset?
# TODO: invert images (255 - image) to get correct coloring (black/white)

dataset_normalize_image <- function(ds, v1=TRUE) {
  dataset_map(ds, function(x) {
    img <- tf$cast(x$image, tf$float32)
    
    normed <- img / tf$norm(img)
    
    # Version 1
    img <- if (v1) 
      (img - tf$math$reduce_mean(img)) / tf$math$reduce_std(img)
    else  
      # Version 2
      x$image <- (255 - img) / 255
    
    x$image <- img # tf$math$abs(img)
    
    x
  })
}


dataset_normalize_image_v2 <- function(ds) {
  dataset_map(ds, function(x) {
    img <- tf$cast(x$image, tf$float32)
    
    x$image <- 
      (img - tf$reduce_min(img)) / (tf$reduce_max(img) - tf$reduce_min(img))
    
    x
  })
}

ds <- ds_raw %>% dataset_normalize_image_v2()
nb <- next_batch(ds)
sess <- tf$Session()
b <- sess$run(nb)

library("jpeg")
# jj <- readJPEG("myfile.jpg",native=TRUE)
plot(0:1,0:1,type="n",ann=FALSE,axes=FALSE)
rasterImage(b$image,0,0,1,1)



library(OpenImageR)
image_augmentation <- function(image) {
  browser()
  filename <- "_image"
  writeImage(image, paste0(filename, ".jpg"))
  
  outfn.prefix=sub(".jpg$|.jpeg$|.png$|.tiff$", "", filename, ignore.case = TRUE)
  
  horizontal_flipped_image=flipImage(image, mode = "horizontal")
  writeImage(horizontal_flipped_image, paste(outfn.prefix, "_horizontal_flipped.jpg",sep=""))
  
  vertical_flipped_image=flipImage(image, mode = "vertical")
  writeImage(vertical_flipped_image, paste(outfn.prefix, "_vertical_flipped.jpg",sep=""))
  
  # randomly rotation
  rotated_image=rotateImage(image, angle = sample(1:359,1))
  writeImage(rotated_image, paste(outfn.prefix, "_rotated.jpg",sep=""))
  
  # image cropping, the cropping dimension can be changed.
  cropped_image=cropImage(image, dim(image)[1]*0.62, dim(image)[2]*0.62)
  writeImage(cropped_image, paste(outfn.prefix, "_cropped.jpg",sep=""))
  
  # ZCA whitening
  zcawhitening_image=ZCAwhiten(image, k = 100, epsilon = 0.1)
  writeImage(zcawhitening_image, paste(outfn.prefix, "_zcawhitening.jpg",sep=""))
  
} 

image_augmentation(b$image)


ds_raw %>% dataset_map(function(x) {
  browser()
  x$image <- image_augmentation(x$image)
})

nb <- next_batch(ds_raw)
sess <- tf$Session()
b <- sess$run(nb)
img <- b$image

