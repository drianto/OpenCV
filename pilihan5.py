import cv2

image = cv2.imread('toyotaInnovaZenix.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

keypoints, descriptors = cv2.ORB_create().detectAndCompute(gray_image, None)

output_image = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))

cv2.imshow('Picture ORB', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
