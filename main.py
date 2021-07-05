import cv2, pickle
import numpy as np
from facef import detection, extraction, distance

def main():
    data = pickle.loads(open('data.pickle', "rb").read())
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            rbg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = detection.ssd_detect(rbg_frame)
            # boxes = detection.dlib_hog_detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # boxes,_ = detection.mtcnn_detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for b in boxes:
                cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), (0,255,0),thickness=2)
                try:
                    emb = extraction.facenet_ext(rbg_frame[b[1]:b[3], b[0]:b[2], :])
                except:
                    print(b)
                    break
                dis = distance.one2many_encoding(emb, data['emb'])
                # print(dis)
                if min(dis)<8:
                    name = data['name'][np.argmin(dis)]
                    frame = cv2.putText(frame, name, (b[0],b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.imshow('Frame',frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

if __name__ == '__main__':
    main()