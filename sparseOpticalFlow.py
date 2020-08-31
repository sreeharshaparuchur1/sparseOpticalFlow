import numpy as np
import cv2

cap = cv2.VideoCapture('./externals/slowTraffic.mp4')

ST_parameters = dict( maxCorners = 50,
                       qualityLevel = 0.2,
                       minDistance = 10,
                       blockSize = 7 )

#Parameters for the ShiTomasi corner detection

#maxCorners gives an upper bound on the number of edges to track in the video.

#qualityLevel dictates how many points are tracked in the video, a lower number means more points
#but a higher chance of interference with other objects in the scene.

#minDistance is the distance between two points in pixels belonging to edges that are to be tracked.

#A higher blockSize may result in wrong tracking of an edge as the corner detection algorithm searches in a larger area.



LK_parameters = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#Parameters for the Lucas-Kanade optical flow.

#Random colours for the vector flow trackign objects.
colour = np.random.randint(25,250,(50,3))

ret, previousFrame = cap.read()
previousFrame_grey = cv2.cvtColor(previousFrame, cv2.COLOR_BGR2GRAY)
previous = cv2.goodFeaturesToTrack(previousFrame_grey, mask = None, **ST_parameters)
#Takes the first frame and finds edges in it.

mask = np.zeros_like(previousFrame)
#Creates a mask image for drawing purposes
#Returns a zero array of the dimensions of the input.

while(cap.isOpened()):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Reads the frame in greyscale to reveal pixel intensities

    next, status, err = cv2.calcOpticalFlowPyrLK(previousFrame_grey, frame_gray, previous, None, **LK_parameters)
    #Calculates the optical flow

    #Selects good points
    goodFeatures_next = next[status == 1]
    #For the next position
    goodFeatures_old = previous[status == 1]
    #For the previous position

    #Displays the track followed by each point.
    for i,(new,old) in enumerate(zip(goodFeatures_next,goodFeatures_old)):
        a,b = new.ravel()
        #Flattened (x,y) coordinates for teh new points of interest.
        c,d = old.ravel()
        #Flattened (x,y) coordinates for the old points of interest.
        mask = cv2.line(mask, (a,b),(c,d), colour[i].tolist(), 2)
        #Connects the old and new points with a random colour.
        frame = cv2.circle(frame,(a,b),5,colour[i].tolist(),1)
        #Highlights the current edge position as a circle/
    output = cv2.add(frame,mask)

    cv2.imshow('Sparse optical flow',output)

    if cv2.waitKey(30) == 27:
        cv2.destroyAllWindows()
        cap.release()
    #Hit escape to quit the optical flow frame
    #Each frame is read in intervals of 30 milliseconds

    #Updates the previous frame and points.
    previousFrame_grey = frame_gray.copy()
    previous = goodFeatures_next.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
