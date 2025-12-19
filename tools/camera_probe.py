import cv2

for dev in ['/dev/video0', '/dev/video2', 0, 2]:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    print(f'Trying {dev!r}: opened', cap.isOpened())
    ret, frame = cap.read()
    print('  read', ret, frame is not None)
    if ret and frame is not None:
        print('  shape', frame.shape)
        print('  mean', frame.mean())
    cap.release()
