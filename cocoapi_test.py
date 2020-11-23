from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import json
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

# draw mask/bbox manually using polygons
def drawPolygon(image, points):
    pts = np.array(points, np.int32)
    # print(pts)
    pts = pts.reshape((-1, 1, 2))

    # setting polygon drawing 
    isClosed = True
    color = (255, 0, 0) # Red color for border
    thickness = 2 # Line thickness of 2 px 
    image = cv2.polylines(image, [pts], isClosed, color, thickness)
    
    return image

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


output_json = []
for i in range(5):
    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person']);
    imgIds = coco.getImgIds(catIds=catIds);
    # imgIds = coco.getImgIds(imgIds = [431]) # imgIds = [436]
    selected_imgId = imgIds[np.random.randint(0,len(imgIds))]
    print('selected_imgId:', selected_imgId)
    img = coco.loadImgs(selected_imgId)[0]

    # load image
    # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    # use url to load image
    I = io.imread(img['coco_url'])

    # load and display instance annotations
    plt.imshow(I); plt.axis('on')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
    anns = coco.loadAnns(annIds)
    # print(anns[0]) # all annotations
    # print('segmentation:', anns[0]['segmentation'])
    # print('bbox:', anns[0]['bbox'])

    # draw bounding box
    # [x,y,w,h] = anns[0]['bbox']
    # cv2.rectangle(I, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 1)

    # show coco mask
    # coco.showAnns(anns)

    # show original image
    # plt.imshow(I)
    # plt.show()


    for i in range(5):
        # rotate_angle = 45 # counter clockwise
        rotate_angle = np.random.randint(low=1,high=360)
        print('rotate_angle:', rotate_angle)
        
        # ========== calculate the points of mask after rotation ==========
        # get the mask from segmentation
        mask = []
        for i in range(0, len(anns[0]['segmentation'][0]), 2):
            point_x = anns[0]['segmentation'][0][i]
            point_y = anns[0]['segmentation'][0][i+1]
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

        max_x, min_x, max_y, min_y = getMaxMin(transformed_mask)

        rotated_img = rotateImage(I, rotate_angle)

        transformed_bbox_endpoint = [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]
        transformed_bbox = [min_x, min_y, max_x-min_x, max_y-min_y] # [x,y,w,h]
        print('new bbox (in format [x,y,width,height]):', transformed_bbox)


        # drawPolygon(rotated_img, transformed_mask) # draw new mask
        # drawPolygon(rotated_img, transformed_bbox_endpoint) # draw new bbox

        # show rotated image
        # plt.imshow(rotated_img)
        # plt.show()

        saved_img = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./train2014_rotated/{imgId}_{angle}.jpg'.format(imgId=selected_imgId, angle=rotate_angle), saved_img)

        output_json.append({"imgId": selected_imgId, "rotate_angle": rotate_angle, "rotate_bbox": transformed_bbox})


# print(json.dumps(output_json, indent=4))
with open('RotatedImages.json', 'w', encoding='utf8') as fp:
    json.dump(output_json, fp, indent=4)