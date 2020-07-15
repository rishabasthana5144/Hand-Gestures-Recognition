##-----------------------------------------------------------------------------------------
## Hand Gesture Recognition
##-----------------------------------------------------------------------------------------

## Importing libraries
import cv2
import numpy as np
import math

# opening Video Live Stream
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)


while True:
    
    # Try block to encounter the error of no hand detetction
    try:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Creating region of interest and bounding box around the hand
        x_coord, y_coord = 250, 100
        width, height = 300, 300
        roi = img[y_coord:(y_coord+height),x_coord:(x_coord+width)]
        cv2.rectangle(img, (x_coord, y_coord), (x_coord+width, y_coord+height), (0, 255, 0), 0)

        # Converting the image to HSV image
        imgHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Creating mask for skin color detection & contour formation
        lower = np.array([0, 32, 64], dtype=np.uint8)
        upper = np.array([34, 255, 255],dtype=np.uint8)

        # Extracting skin color images
        mask = cv2.inRange(imgHSV, lower, upper)

        # Erosions and dilations to get rid of enough noise
        # First -  Erosion
        # Creating kernel
        kernel = np.ones((3, 3), np.uint8)

        #Second - Dilation
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Blurring the image to reduce noise
        mask = cv2.GaussianBlur(mask, (5,5),100)

        # Finding the contours around palm
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        # Approximating contour by removing the noise
        epsilon = 2.5 * cv2.arcLength(cnt, True) #0.05
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Convex hull around the hand
        hull = []
        hull = cv2.convexHull(cnt, False)

        # Drawing contours and hull around the palm - Optional
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
        cv2.drawContours(img, hull, -1, (0, 0, 255), 3)

        areahull = cv2.contourArea(hull)
        palmarea = cv2.contourArea(cnt)
  

        # Area ratio
        arearatio = ((areahull - palmarea) / palmarea) * 100

        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        
        # Counting Fingers 
        finger_cnt = 0
        for i in range(defects.shape[0]):  # calculate the angle

            s, e, f, d = defects[i][0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
            
            if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                finger_cnt += 1
                # draw lines around hand
                cv2.line(roi, start, end, [0, 255, 0], 2)
        finger_cnt = finger_cnt+1

        # Printing finger count on screen
        if finger_cnt == 1:
            if arearatio < 5:
                cv2.putText(img,"Show fingers to count",(0,50), cv2.FONT_ITALIC, 1, (0, 0, 255), 3, cv2.LINE_AA)
            if arearatio >= 20:
                cv2.putText(img,"1",(0,50), cv2.FONT_ITALIC, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif finger_cnt == 2:
            cv2.putText(img, "2", (0, 50), cv2.FONT_ITALIC, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif finger_cnt ==3:
            cv2.putText(img, "3", (0, 50), cv2.FONT_ITALIC, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif finger_cnt ==4:
            cv2.putText(img, "4", (0, 50), cv2.FONT_ITALIC, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif finger_cnt ==5:
            cv2.putText(img, "5", (0, 50), cv2.FONT_ITALIC, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif finger_cnt ==6:
            cv2.putText(img, "6", (0, 50), cv2.FONT_ITALIC, 2, (0, 0, 255), 3, cv2.LINE_AA)
        # else:
        #      cv2.putText(img, "Reposition", (0, 50), cv2.FONT_ITALIC, 2, (0, 0, 255), 3, cv2.LINE_AA)

        print(finger_cnt)

        cv2.imshow("Masked", mask)
        cv2.imshow("Original", img)
    except:
        pass

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


