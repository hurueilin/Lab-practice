from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import cv2

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

def getMaxMin(transformed_mask):
    max_x = min_x = transformed_mask[0][0] 
    max_y = min_y = transformed_mask[0][1] 
    for i in range(len(transformed_mask)):
        if transformed_mask[i][0] > max_x:
            max_x = transformed_mask[i][0]
        if transformed_mask[i][0] < min_x:
            min_x = transformed_mask[i][0]
        if transformed_mask[i][1] > max_y:
            max_y = transformed_mask[i][1]
        if transformed_mask[i][1] < min_y:
            min_y = transformed_mask[i][1]
    return max_x, min_x, max_y, min_y



catIds = coco.getCatIds(catNms=['person']);
imgIds = coco.getImgIds(catIds=catIds);

def Rotatedbbox(imgId, rotate_angle):
    selected_imgId = imgId
    print('selected_imgId:', selected_imgId)
    img = coco.loadImgs(selected_imgId)[0]

    # load image
    # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    # use url to load image
    I = io.imread(img['coco_url'])

    # load and display instance annotations
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
    anns = coco.loadAnns(annIds)
    print('number of objects:', len(anns))



    maxmin_list = []
    # for each object in image
    for k in range(len(anns)):
        # ignore the iscrowd=1 object
        if(anns[k]['iscrowd'] == 1):
            continue
        # ========== calculate the points of mask after rotation ==========
        # get the mask from segmentation
        mask = []
        for m in range(0, len(anns[k]['segmentation'][0]), 2):
            point_x = anns[k]['segmentation'][0][m]
            point_y = anns[k]['segmentation'][0][m+1]
            mask.append([point_x, point_y])
        # print('mask:', mask)

        # pad 1 for Homogeneous
        ones = np.ones(shape=(len(mask), 1))
        mask_addingOne = np.hstack([mask, ones])

        # calculate RotationMatrix and transform the points in mask
        rotation_matrix = calRotationMatrix(I, rotate_angle)
        transformed_mask = rotation_matrix.dot(mask_addingOne.T).T
        # print('transformed_mask:', transformed_mask)
        # ========================= end =========================
        x_max, x_min, y_max, y_min = getMaxMin(transformed_mask)
        maxmin_list.append([int(x_min), int(y_min), int(x_max), int(y_max)])

    return maxmin_list


imgId = 144391
rotate_angle = np.random.randint(low=1,high=360)
print('rotate_angle:', rotate_angle)
print(Rotatedbbox(imgId, rotate_angle))