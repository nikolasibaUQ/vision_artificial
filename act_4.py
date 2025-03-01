import cv2
import metodos as mt
import matplotlib.pyplot as plt


defaultOutput = 'act4/resources/'
defaultImage = 'act4/resources/image.png'

mt.delete_noides_closed_image(defaultImage, defaultOutput+'closed.png', True, 3)

mt.delete_noides_opened_image(defaultImage, defaultOutput+'opened.png', True, 3)

mt.gradient_image(defaultImage, defaultOutput+'gradient.png', True, 3)

mt.fill_noise_opened(defaultImage, defaultOutput+'filledopened.png', True, 3)
mt.fill_noise_closed(defaultImage, defaultOutput+'filledclosed.png', True, 3)




#graficate the 3 images with matplotlib


fig, axs = plt.subplots(1, 6, figsize=(15, 5))
axs[0].imshow(cv2.imread(defaultOutput+'closed.png'))
axs[0].set_title('Closed Image')
axs[0].axis('off')
axs[1].imshow(cv2.imread(defaultOutput+'opened.png'))
axs[1].set_title('Opened Image')
axs[1].axis('off')
axs[2].imshow(plt.imread(defaultImage))
axs[2].set_title('Original Image')
axs[2].axis('off')
axs[3].imshow(cv2.imread(defaultOutput+'gradient.png'))
axs[3].set_title('Gradient Image')
axs[3].axis('off')
axs[4].imshow(cv2.imread(defaultOutput+'filledopened.png'))
axs[4].set_title('Filled open  Image')
axs[4].axis('off')
axs[5].imshow(cv2.imread(defaultOutput+'filledclosed.png'))
axs[5].set_title('Filled close Image')
axs[5].axis('off')
plt.tight_layout()
plt.show()




