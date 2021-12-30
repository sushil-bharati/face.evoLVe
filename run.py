from PIL import Image
from applications.align.detector import detect_faces
from applications.align.visualization_utils import show_results
from applications.align.align_trans import get_reference_facial_points, warp_and_crop_face
from util.extract_feature_v1 import extract_feature
from models.ms1m_ir50.model_irse import *
import numpy as np
import os

CROP_SIZE = 112

img = Image.open('/share/project/fair-face-rec/pushkar/FactorVAE/data/CelebA/img_align_celeba/000001.jpg') # modify the image path to yours
print(img.size)
bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
# print(bounding_boxes)
# print(landmarks)
# img_det = show_results(img, bounding_boxes, landmarks) # visualize the results
# img_det.show()

# align face
scale = CROP_SIZE / 112.
reference = get_reference_facial_points(default_square = True) * scale
facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(CROP_SIZE, CROP_SIZE))
img_warped = Image.fromarray(warped_face)
data_root = "/share/project/fair-face-rec/sushil/face.evoLVe/transforms/"
LABEL_IMG = "test_candidate_001"
filePath = os.path.join(data_root, LABEL_IMG)
if not os.path.exists(filePath) : os.makedirs(filePath)
img_warped.save(os.path.join(filePath,"test_transformed.jpg"))

# determine feature

backbone = Backbone(input_size = [112, 122, 3], num_layers=50)
model_root = "/share/project/fair-face-rec/sushil/face.evoLVe/models/ms1m_ir50/backbone_ir50_ms1m_epoch120.pth"
feat = extract_feature(data_root, backbone, model_root)
print(feat.shape)
print(feat)
