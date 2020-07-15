# Hand-Gestures-Recognition

Using python's most famous opencv library to detect the gestures of the hand such as finger counting.

# High Level Steps
Step 1: Open camera and capture the video live stream

Step 2: Define the region of interest on live stream and create a box around it

Step 3: Convert the BGR image to HSV image (Hue, Saturation and Value) and provide range of values to detect hand color out of all available.

Step 4: Erode, Dilate and Blur the image to reduce noise.

Step 5: Detect contour area around hand and convex hull

Step 6: Identify the defects between the fingers (Valley points between two fingers)

Step 7: Calculate the ange between fingers and fingers raised.

Step 8: Display the count on the screen. Voila. You are done 
