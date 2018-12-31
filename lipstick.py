# USAGE
# python lipstick.py --shape-predictor shape_predictor_68_face_landmarks.dat
# press esc to exit
import numpy as np
import cv2
import sys
import dlib
import math
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
from skimage import io

def distance (a, b):
    dis_x = (a.x - b.x)**2
    dis_y = (a.y - b.y)**2
    return math.sqrt(dis_x + dis_y)

def save_points(im, rects):
    path = "shape_predictor_68_face_landmarks.dat"
    predict = dlib.shape_predictor(path)
    landmark = predict(im, rects[0])
    points = []
    dlib_points = []
    for idx, point in enumerate(landmark.parts()):
        if idx >= 48:
            dlib_points.append(point)
            points.append((int(point.x), int(point.y)))
    triangles = [[1, 12, 13], [5, 15, 16], [7, 16, 17], [11, 12, 19]]
    for tri in triangles:
        if distance(dlib_points[14], dlib_points[18]) > 45:
            x = int(0.5 * dlib_points[tri[0]].x + 0.25 * dlib_points[tri[1]].x + 0.25 * dlib_points[tri[2]].x)
            y = int(0.5 * dlib_points[tri[0]].y + 0.25 * dlib_points[tri[1]].y + 0.25 * dlib_points[tri[2]].y)  
        else:
            x = int(0.5 * dlib_points[tri[1]].x + 0.5 * dlib_points[tri[2]].x)
            y = int(0.5 * dlib_points[tri[1]].y + 0.5 * dlib_points[tri[2]].y)
        points.append((x, y))
    
    return points


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    #do dich chuyen tu 3 dinh tam giac toi dinh bounding box
    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

    
def nothing(x):
    pass


if __name__ == '__main__' :
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-r", "--picamera", type=int, default=-1,
        help="whether or not the Raspberry Pi camera should be used")
    args = vars(ap.parse_args())
     
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    time.sleep(2.0)

        
    filename1 = 'g1.jpg'
    alpha = 0.7
    
    # Read images
    img1 = cv2.imread(filename1);
    
    rects1 = detector(img1,1)
    
    points1 = save_points(img1, rects1)
    
    # Convert Mat to float data type
    img1 = np.float32(img1)
    
    cv2.namedWindow('Morphed Face')
    
    # Read triangles from tri.txt
    tri = []
    with open("tri.txt") as file :
        for line in file :
            x,y,z = line.split()
            tri.append((int(x),int(y),int(z))) 

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 800 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        
        
        # detect faces in the frame
        rects = detector(frame, 1)
        if len(rects) > 0:
            points2 = save_points(frame, rects)
            frame = np.float32(frame)
            
            
            imgMorph = frame
            points = points2
            for i in range(0, 5):
                cv2.imshow("Morphed Face", np.uint8(imgMorph))
                
                for tr in tri :
                    x = tr[0]
                    y = tr[1] 
                    z = tr[2] 
                        
                    t1 = [points1[x], points1[y], points1[z]]
                    t2 = [points2[x], points2[y], points2[z]]
                    t = [points[x], points[y], points[z]]
                        
                    # Morph one triangle at a time.
                    morphTriangle(img1, frame, imgMorph, t1, t2, t, alpha)    
                
                key = cv2.waitKey(1) & 0xFF
                # if the `esc` key was pressed, break from the loop
                if key == 27:
                    break
        else:
            cv2.imshow("Morphed Face", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `esc` key was pressed, break from the loop
        if key == 27:
            break
    

    cv2.destroyAllWindows()
