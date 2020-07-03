import cv2
import numpy as np

def func(ar):
    ar = ar.reshape((4,2))
    ar1 = np.zeros((4,2),dtype = np.float32)

    add = ar.sum(1)
    ar1[0] = ar[np.argmin(add)]
    ar1[2] = ar[np.argmax(add)]

    diff = np.diff(ar,axis = 1)
    ar1[1] = ar[np.argmin(diff)]
    ar1[3] = ar[np.argmax(diff)]

    return ar1

"""read document"""
img=cv2.imread("C:/Users/yasha/Desktop/pics/test_image.jpg")
cv2.imshow("Original image",img)

"""resize document"""
img=cv2.resize(img,(1300,800))

imgOr=img.copy()

"""convert bgr to gray scale"""
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray image",imgGray)

"""convert gray image to blurred image (5,5) is kernel size and 0 is sigma"""
imgBlur=cv2.GaussianBlur(imgGray,(5,5),0)
cv2.imshow("Blurred image",imgBlur)

"""determine the edges"""
imgEdge=cv2.Canny(imgBlur,30,50)
cv2.imshow("Canny image",imgEdge)

"""determining contours and returning it as a list using simple chain approximation method"""
contours,hierarchy=cv2.findContours(imgEdge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

"""sort contours and get maximum area contour"""
contours=sorted(contours,key=cv2.contourArea,reverse=True)

"""looping to get the boundary contours of the page"""
for i in contours:
    """extracting a curve i that is closed """
    a=cv2.arcLength(i,True)

    """Approximates a polygonal curve with the specified precision."""
    """i is for curve, 0.02*a is for epsilon value, True is for closed curve"""
    res=cv2.approxPolyDP(i,0.02*a,True)

    if len(res)==4:
        t=res
        break

"""func() finds endpoints of the sheet"""
res=func(t)

"""mapping to 800x800 window"""
pts=np.float32([[0,0],[800,0],[800,800],[0,800]])

imgPers=cv2.getPerspectiveTransform(res,pts)
imgFinal=cv2.warpPerspective(imgOr,imgPers,(800,800))

cv2.imshow("Document Scanned",imgFinal)
cv2.waitKey(0)
