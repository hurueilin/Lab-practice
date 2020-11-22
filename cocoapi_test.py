from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='..'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)

def calRotationMatrix(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return M

def rotateImage(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# draw mask manually using polygons
def drawMask(image, mask):
    pts = np.array(mask, np.int32)
    # print(pts)
    pts = pts.reshape((-1, 1, 2))

    # setting polygon drawing 
    isClosed = True
    color = (255, 0, 0) # Red color for border
    thickness = 2 # Line thickness of 2 px 
    image = cv2.polylines(image, [pts], isClosed, color, thickness)
    
    return image


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person']);
imgIds = coco.getImgIds(catIds=catIds);
imgIds = coco.getImgIds(imgIds = [436]) # imgIds = [436]
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# load image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])

# load and display instance annotations
plt.imshow(I); plt.axis('on')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
anns = coco.loadAnns(annIds)
# print(anns[0]) # all annotations
# print('segmentation: ', anns[0]['segmentation'])

# draw bounding box
# print('bbox: ', anns[0]['bbox'])
# [x,y,w,h] = anns[0]['bbox']
# cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 1)

# show coco mask
# coco.showAnns(anns)

plt.imshow(I)
plt.show()


# ========== calculate the points of mask after rotation ==========
# get the mask from segmentation
mask = []
for i in range(0, len(anns[0]['segmentation'][0]), 2):
	point_x = anns[0]['segmentation'][0][i]
	point_y = anns[0]['segmentation'][0][i+1]
	mask.append([point_x, point_y])
print('mask: ', mask)

# pad 1 for Homogeneous
ones = np.ones(shape=(len(mask), 1))
mask_addingOne = np.hstack([mask, ones])
print(mask_addingOne)

# calculate RotationMatrix and transform the points in mask
rotation_matrix = calRotationMatrix(I, 20)
transformed_mask = rotation_matrix.dot(mask_addingOne.T).T
print('transformed_mask:', transformed_mask)
# ========================= end =========================


rotated_img = rotateImage(I, 20)
drawMask(rotated_img, transformed_mask) # draw transformed mask
plt.imshow(rotated_img)
plt.show()