from facef import detection, distance, extraction
import cv2, pickle, glob, tqdm
import numpy as np

def train():
    list_folder = glob.glob('data/*')
    data = {}
    names = []
    embs = []
    for i in tqdm.tqdm(range(len(list_folder))):
        folder = list_folder[i]
        name = folder.split('/')[-1]
        list_file = glob.glob(folder + '/*')
        for f in list_file:
            img = cv2.imread(f)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            box = detection.ssd_detect(rgb)[0]
            emb = extraction.facenet_ext(rgb[box[1]:box[3], box[0]:box[2], :])
            names.append(name)
            embs.append(emb)
    data['name'] = np.array(names)
    data['emb'] = np.array(embs)
    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)

train()