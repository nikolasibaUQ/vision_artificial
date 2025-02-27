import metodos as mt


defaultImage = 'lab1/resources/principal_image.jpg'
defaultOutput = 'lab1/resources/'


# generate original vectror image

original_media = mt.get_media_image(defaultImage)
original_desviation = mt.get_desviation_image(defaultImage)
original_mean = mt.get_mean_image(defaultImage)
original_mode = mt.get_mode_image(defaultImage)

# generate vector image with noise

original_vector = [original_media,
                   original_desviation, original_mean, original_mode]


# generate images with methods

mt.gray_image(defaultImage, defaultOutput + 'gray_image.jpg', True)
mt.binary_image(defaultImage, defaultOutput + 'binary_image.jpg', True)
mt.convert_gray_scale(defaultImage, defaultOutput +
                      'gray_scale_image.jpg', True)
mt.convert_RGB(defaultImage, defaultOutput + 'rgb_image.jpg', True)


# generate data for each image

gray_media = mt.get_media_image(defaultOutput + 'gray_image.jpg')
gray_desviation = mt.get_desviation_image(defaultOutput + 'gray_image.jpg')
gray_mean = mt.get_mean_image(defaultOutput + 'gray_image.jpg')
gray_mode = mt.get_mode_image(defaultOutput + 'gray_image.jpg')

gray_vector = [gray_media, gray_desviation, gray_mean, gray_mode]

binary_media = mt.get_media_image(defaultOutput + 'binary_image.jpg')
binary_desviation = mt.get_desviation_image(defaultOutput + 'binary_image.jpg')
binary_mean = mt.get_mean_image(defaultOutput + 'binary_image.jpg')
binary_mode = mt.get_mode_image(defaultOutput + 'binary_image.jpg')

binary_vector = [binary_media, binary_desviation, binary_mean, binary_mode]

gray_scale_media = mt.get_media_image(defaultOutput + 'gray_scale_image.jpg')
gray_scale_desviation = mt.get_desviation_image(
    defaultOutput + 'gray_scale_image.jpg')
gray_scale_mean = mt.get_mean_image(defaultOutput + 'gray_scale_image.jpg')
gray_scale_mode = mt.get_mode_image(defaultOutput + 'gray_scale_image.jpg')

gray_scale_vector = [gray_scale_media,
                     gray_scale_desviation, gray_scale_mean, gray_scale_mode]

rgb_media = mt.get_media_image(defaultOutput + 'rgb_image.jpg')
rgb_desviation = mt.get_desviation_image(defaultOutput + 'rgb_image.jpg')
rgb_mean = mt.get_mean_image(defaultOutput + 'rgb_image.jpg')
rgb_mode = mt.get_mode_image(defaultOutput + 'rgb_image.jpg')

rgb_vector = [rgb_media, rgb_desviation, rgb_mean, rgb_mode]

# now i need rotate the image in 180 degrees and save it

mt.rotate_image(defaultImage, defaultOutput + 'rotate_image.jpg', True, 180)

# generate filter image rotate

rotate_media = mt.get_media_image(defaultOutput + 'rotate_image.jpg')
rotate_desviation = mt.get_desviation_image(defaultOutput + 'rotate_image.jpg')
rotate_mean = mt.get_mean_image(defaultOutput + 'rotate_image.jpg')
rotate_mode = mt.get_mode_image(defaultOutput + 'rotate_image.jpg')

rotate_vector = [rotate_media, rotate_desviation, rotate_mean, rotate_mode]

# generate images rotate with filters

mt.gray_image(defaultOutput + 'rotate_image.jpg',
              defaultOutput + 'gray_rotate_image.jpg', True)
mt.binary_image(defaultOutput + 'rotate_image.jpg',
                defaultOutput + 'binary_rotate_image.jpg', True)
mt.convert_gray_scale(defaultOutput + 'rotate_image.jpg',
                      defaultOutput + 'gray_scale_rotate_image.jpg', True)
mt.convert_RGB(defaultOutput + 'rotate_image.jpg',
               defaultOutput + 'rgb_rotate_image.jpg', True)

# generate data for each image

gray_rotate_media = mt.get_media_image(defaultOutput + 'gray_rotate_image.jpg')
gray_rotate_desviation = mt.get_desviation_image(
    defaultOutput + 'gray_rotate_image.jpg')
gray_rotate_mean = mt.get_mean_image(defaultOutput + 'gray_rotate_image.jpg')
gray_rotate_mode = mt.get_mode_image(defaultOutput + 'gray_rotate_image.jpg')

gray_rotate_vector = [gray_rotate_media,
                      gray_rotate_desviation, gray_rotate_mean, gray_rotate_mode]

binary_rotate_media = mt.get_media_image(
    defaultOutput + 'binary_rotate_image.jpg')
binary_rotate_desviation = mt.get_desviation_image(
    defaultOutput + 'binary_rotate_image.jpg')
binary_rotate_mean = mt.get_mean_image(
    defaultOutput + 'binary_rotate_image.jpg')
binary_rotate_mode = mt.get_mode_image(
    defaultOutput + 'binary_rotate_image.jpg')

binary_rotate_vector = [binary_rotate_media,
                        binary_rotate_desviation, binary_rotate_mean, binary_rotate_mode]

gray_scale_rotate_media = mt.get_media_image(
    defaultOutput + 'gray_scale_rotate_image.jpg')
gray_scale_rotate_desviation = mt.get_desviation_image(
    defaultOutput + 'gray_scale_rotate_image.jpg')
gray_scale_rotate_mean = mt.get_mean_image(
    defaultOutput + 'gray_scale_rotate_image.jpg')
gray_scale_rotate_mode = mt.get_mode_image(
    defaultOutput + 'gray_scale_rotate_image.jpg')

gray_scale_rotate_vector = [gray_scale_rotate_media,
                            gray_scale_rotate_desviation, gray_scale_rotate_mean, gray_scale_rotate_mode]

rgb_rotate_media = mt.get_media_image(defaultOutput + 'rgb_rotate_image.jpg')
rgb_rotate_desviation = mt.get_desviation_image(
    defaultOutput + 'rgb_rotate_image.jpg')
rgb_rotate_mean = mt.get_mean_image(defaultOutput + 'rgb_rotate_image.jpg')
rgb_rotate_mode = mt.get_mode_image(defaultOutput + 'rgb_rotate_image.jpg')

rgb_rotate_vector = [rgb_rotate_media,
                     rgb_rotate_desviation, rgb_rotate_mean, rgb_rotate_mode]


print('Original Image')
print(*original_vector)
print('Rotate Image')
print(*rotate_vector)


print('Gray Image')
print(*gray_vector)
print('Gray Rotate Image')
print(*gray_rotate_vector)


print('Binary Image')
print(*binary_vector)
print('Binary Rotate Image')
print(*binary_rotate_vector)


print('Gray Scale Image')
print(*gray_scale_vector)
print('Gray Scale Rotate Image')
print(*gray_scale_rotate_vector)


print('RGB Image')
print(*rgb_vector)
print('RGB Rotate Image')
print(*rgb_rotate_vector)
