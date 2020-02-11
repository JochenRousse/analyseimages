import numpy as np
from keras.preprocessing.image import image
from keras.models import load_model
import os
import shutil
from sklearn.metrics import classification_report, confusion_matrix


def reports(y_pred, y_true, classs_names):
    classification = classification_report(y_true, y_pred, target_names=classs_names)
    confusion = confusion_matrix(y_true, y_pred)

    return classification, confusion


model = load_model('CorelDB_model.h5')

img_width, img_height = 120, 120

img_dir = "./data/src"
resu_dir = "./data/result"

if not os.path.exists(resu_dir):
    os.mkdir(resu_dir)

nb_test_samples = sum([len(files) for r, d, files in os.walk(img_dir)])

batch_holder = np.zeros((nb_test_samples, img_width, img_height, 3))
y_true = np.zeros(nb_test_samples)

class_names = next(os.walk(img_dir))[1]

i = 0
for dirpath, dirnames, filenames in os.walk(img_dir):
    for imgnm in filenames:
        img = image.load_img(os.path.join(dirpath, imgnm), target_size=(img_width, img_height))
        batch_holder[i, :] = img
        y_true[i] = int(class_names.index(os.path.relpath(dirpath, img_dir)))
        i = i + 1

y_pred = model.predict_classes(batch_holder)

classification, confusion = reports(y_pred, y_true, class_names)

print(classification)
print(confusion)

for dirpath, dirnames, filenames in os.walk(img_dir):
    structure = os.path.join(resu_dir, os.path.relpath(dirpath, img_dir))
    if not os.path.isdir(structure):
        os.mkdir(structure)
    else:
        print("Folder already exists")

i = 0
for dirpath, dirnames, filenames in os.walk(img_dir):
    structure = os.path.join(resu_dir, os.path.relpath(dirpath, img_dir))
    for imgnm in filenames:
        shutil.copy(dirpath + '/' + imgnm, resu_dir + '/' + class_names[y_pred[i]] + '/' + imgnm)
        i = i + 1
