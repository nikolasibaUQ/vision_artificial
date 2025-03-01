import cv2
import metodos as mt
import matplotlib.pyplot as plt

# 1. read the images
defaultOutput = 'act4/resources/'
img2 = 'act4/resources/image.png'
img1 = 'act4/resources/image_2.png'
img3 = 'act4/resources/image_3.png'
img4 = 'act4/resources/image_4.png'

# 2. apply the morphological operations
# a. delete the noise with closing
mt.delete_noides_closed_image(img1, defaultOutput+'closed_1.png',True)
mt.delete_noides_closed_image(img2, defaultOutput+'closed_2.png',True)
mt.delete_noides_closed_image(img3, defaultOutput+'closed_3.png',True)
mt.delete_noides_closed_image(img4, defaultOutput+'closed_4.png',True)
# b. delete the noise with opening
mt.delete_noides_opened_image(img1, defaultOutput+'opened_1.png',True)
mt.delete_noides_opened_image(img2, defaultOutput+'opened_2.png',True)
mt.delete_noides_opened_image(img3, defaultOutput+'opened_3.png',True)
mt.delete_noides_opened_image(img4, defaultOutput+'opened_4.png',True)
# c. apply the gradient
mt.gradient_image(img1, defaultOutput+'gradient_1.png',True)
mt.gradient_image(img2, defaultOutput+'gradient_2.png',True)
mt.gradient_image(img3, defaultOutput+'gradient_3.png',True)
mt.gradient_image(img4, defaultOutput+'gradient_4.png',True)
# d. fill the noise with opening
mt.fill_noise_opened(img1, defaultOutput+'fill_noise_opened_1.png',True)
mt.fill_noise_opened(img2, defaultOutput+'fill_noise_opened_2.png',True)
mt.fill_noise_opened(img3, defaultOutput+'fill_noise_opened_3.png',True)
mt.fill_noise_opened(img4, defaultOutput+'fill_noise_opened_4.png',True)
# e. fill the noise with closing
mt.fill_noise_closed(img1, defaultOutput+'fill_noise_closed_1.png',True)
mt.fill_noise_closed(img2, defaultOutput+'fill_noise_closed_2.png',True)
mt.fill_noise_closed(img3, defaultOutput+'fill_noise_closed_3.png',True)
mt.fill_noise_closed(img4, defaultOutput+'fill_noise_closed_4.png',True)

# graficate the  images with matplotlib
# 3. show the images with matplotlib
fig, axs = plt.subplots(6, 4, figsize=(15, 5))
 #compare with original images
axs[0,0].imshow(cv2.imread(img1))
axs[0,0].set_title('Original Image #1')
axs[0,0].axis('off')
axs[0,1].imshow(cv2.imread(img2))
axs[0,1].set_title('Original Image #2')
axs[0,1].axis('off')
axs[0,2].imshow(cv2.imread(img3))
axs[0,2].set_title('Original Image #3')
axs[0,2].axis('off')
axs[0,3].imshow(cv2.imread(img4))
axs[0,3].set_title('Original Image #4')
axs[0,3].axis('off')


axs[1,0].imshow(cv2.imread(defaultOutput+'closed_1.png'))
axs[1,0].set_title('Closed Image #1')
axs[1,0].axis('off')
axs[1,1].imshow(cv2.imread(defaultOutput+'closed_2.png'))
axs[1,1].set_title('Closed Image #2')
axs[1,1].axis('off')
axs[1,2].imshow(cv2.imread(defaultOutput+'closed_3.png'))
axs[1,2].set_title('Closed Image #3')
axs[1,2].axis('off')
axs[1,3].imshow(cv2.imread(defaultOutput+'closed_4.png'))
axs[1,3].set_title('Closed Image #4')
axs[1,3].axis('off')

axs[2,0].imshow(cv2.imread(defaultOutput+'opened_1.png'))
axs[2,0].set_title('Opened Image #1')
axs[2,0].axis('off')
axs[2,1].imshow(cv2.imread(defaultOutput+'opened_2.png'))
axs[2,1].set_title('Opened Image #2')
axs[2,1].axis('off')
axs[2,2].imshow(cv2.imread(defaultOutput+'opened_3.png'))
axs[2,2].set_title('Opened Image #3')
axs[2,2].axis('off')
axs[2,3].imshow(cv2.imread(defaultOutput+'opened_4.png'))
axs[2,3].set_title('Opened Image #4')
axs[2,3].axis('off')

axs[3,0].imshow(cv2.imread(defaultOutput+'gradient_1.png'))
axs[3,0].set_title('Gradient Image #1')
axs[3,0].axis('off')
axs[3,1].imshow(cv2.imread(defaultOutput+'gradient_2.png'))
axs[3,1].set_title('Gradient Image #2')
axs[3,1].axis('off')
axs[3,2].imshow(cv2.imread(defaultOutput+'gradient_3.png'))
axs[3,2].set_title('Gradient Image #3')
axs[3,2].axis('off')
axs[3,3].imshow(cv2.imread(defaultOutput+'gradient_4.png'))
axs[3,3].set_title('Gradient Image #4')
axs[3,3].axis('off')

axs[4,0].imshow(cv2.imread(defaultOutput+'fill_noise_opened_1.png'))
axs[4,0].set_title('Fill Noise Opened Image #1')
axs[4,0].axis('off')
axs[4,1].imshow(cv2.imread(defaultOutput+'fill_noise_opened_2.png'))
axs[4,1].set_title('Fill Noise Opened Image #2')
axs[4,1].axis('off')
axs[4,2].imshow(cv2.imread(defaultOutput+'fill_noise_opened_3.png'))
axs[4,2].set_title('Fill Noise Opened Image #3')
axs[4,2].axis('off')
axs[4,3].imshow(cv2.imread(defaultOutput+'fill_noise_opened_4.png'))
axs[4,3].set_title('Fill Noise Opened Image #4')
axs[4,3].axis('off')

axs[5,0].imshow(cv2.imread(defaultOutput+'fill_noise_closed_1.png'))
axs[5,0].set_title('Fill Noise Closed Image #1')
axs[5,0].axis('off')
axs[5,1].imshow(cv2.imread(defaultOutput+'fill_noise_closed_2.png'))
axs[5,1].set_title('Fill Noise Closed Image #2')
axs[5,1].axis('off')
axs[5,2].imshow(cv2.imread(defaultOutput+'fill_noise_closed_3.png'))
axs[5,2].set_title('Fill Noise Closed Image #3')
axs[5,2].axis('off')
axs[5,3].imshow(cv2.imread(defaultOutput+'fill_noise_closed_4.png'))
axs[5,3].set_title('Fill Noise Closed Image #4')
axs[5,3].axis('off')


plt.tight_layout()
plt.show()

