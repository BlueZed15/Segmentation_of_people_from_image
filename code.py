import cv2, numpy as ny
draw, dx, dy, bound, method,cnt = 0, -1, -1, (), 0, 0

def detect(ima):
    ob = cv2.dnn.readNet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
    im = cv2.dnn.blobFromImage(ima, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    ob.setInput(im)
    lay = ob.getLayerNames()
    lay2 = [lay[i - 1] for i in ob.getUnconnectedOutLayers()]
    out = ob.forward(lay2)
    bound= []
    boun=[]
    confis = []
    h, w = ima.shape[:2]
    for a in out:
        for det in a:
            prob = det[5:]
            confid = prob[ny.argmax(prob)]
            if confid > 0.6:
                box = det[:4] * (ny.array([w, h, w, h]))
                (cx, cy, wi, hi) = box.astype("int")
                x, y = int(cx - (wi / 2)), int(cy - (hi / 2))
                confis.append(float(confid))
                bound.append([x, y, int(wi), int(hi)])

    index = cv2.dnn.NMSBoxes(bound, confis, 0.25, 0.5)
    if len(index) > 0:
        for i in index:
            rect = bound[i]
            xe, ye = int(rect[0] + rect[2]), int(rect[1] + rect[3])
            cv2.rectangle(clone2, (int(rect[0]), int(rect[1])), (xe, ye), (0, 255, 0), 1)
            boun.append(bound[i])
    return boun, clone2


def seg(image, bound2, mask, fore, back):
    global clone,cnt
    if ch.lower() in['a','r']:
        cv2.grabCut(image, mask, bound2, back, fore, 5, cv2.GC_INIT_WITH_RECT)
    else:
        cv2.grabCut(image, mask, None, back, fore, 5, cv2.GC_INIT_WITH_MASK)
    print('success1')
    mask2 = ny.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    clone = image * mask2[:, :, ny.newaxis]
    cv2.imshow('fin', clone)
    cv2.waitKey(0)

def patimg(event, x, y, flags, param):
    global draw, dx, dy, clone2,bound, method, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        draw, dx, dy = 1, x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw == 1 and method == 0:
            clone2 = ima.copy()
            cv2.rectangle(clone2, (dx, dy), (x, y), (0, 0, 255), 2)
        elif draw == 1 and method == 1:
            cv2.line(clone2, (dx, dy), (x, y), (0, 0, 0), 4)
            cv2.line(mask, (dx, dy), (x, y), (0, 0, 0), 4)
        elif draw == 1 and method == 2:
            cv2.line(clone2, (dx, dy), (x, y), (255, 255, 255), 4)
            cv2.line(mask, (dx, dy), (x, y), (255, 255, 255), 4)
    elif event == cv2.EVENT_LBUTTONUP:
        draw = 0
        if ch.lower() == 'r':
            cv2.rectangle(clone2, (dx, dy), (x, y), (0, 0, 255), 2)
            bound = (dx, dy, x - dx, y - dy)


# _main_
ima = cv2.imread(r"C:\Users\Rohith\Downloads\20220814_155744.jpg")
ima = cv2.resize(ima, (560, 560), 0)
ima = cv2.medianBlur(ima, 1)
clone2=clone = ima.copy()
mask = ny.zeros(ima.shape[:2], ny.uint8)
newmask = ny.zeros(ima.shape[:2], ny.uint8)
fore = back = ny.zeros((1, 65))
bound, clone2 = detect(clone2)
cv2.imshow('image_with_identified_object',clone2)
cv2.waitKey(0)

ch = input("If you want automatic segmentation, press 'a'/'A'\nIf you want to manually highlight "
           "foreground/background,press 'm'/'M'\nIf you want to draw rectangle manually,press 'r'/'R'...")
if ch.lower() == 'm':
    print('\nPress 1 to highlight background or Press 2 to highlight background...')
    cv2.namedWindow('Highlighted_image')
    cv2.setMouseCallback('Highlighted_image', patimg)
    while True:
        cv2.imshow('Highlighted_image',clone2)
        k = cv2.waitKey(1)
        if k == ord('1'):
            method = 1
            print('You are highlighting background')
        elif k == ord('2'):
            method = 2
            print('You are highlighting foreground')
        elif k == ord('q'):
            break
    newmask[mask == 0] = 0
    newmask[mask == 255] = 1
    seg(ima, bound, newmask, fore, back)
elif ch.lower() == 'r':
    cv2.namedWindow('Highlighted_image')
    cv2.setMouseCallback('Highlighted_image', patimg)
    while True:
        cv2.imshow('Highlighted_image', clone2)
        if cv2.waitKey(1) == ord('q'):
            break
    seg(ima, bound, mask, fore, back)
else:
    print(bound)
    for i in bound:
        seg(ima,i,mask,fore,back)

cv2.destroyAllWindows()
