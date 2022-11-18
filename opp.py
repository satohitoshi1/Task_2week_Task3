import cv2
import numpy as np

# 画像の読み込み
eye = cv2.imread("images/eye_img.png", -1)
face = cv2.imread("images/face_img.png")

# eyeの透過処理その1
eye[:, :, 3] = np.where(np.all(eye == 255, axis=-1), 0, 255)
cv2.imwrite("images/clear_eye.png", eye)

clear_eye = cv2.imread("images/clear_eye.png")

#  白色部分に対応するマスク画像を生成 透過処理その2 色が違う言われる
# eye_mask = np.all(eye[:, :, :] == [255, 255, 255], axis=-1)
#  元画像をBGR形式からBGRA形式に変換
# eye_dst = cv2.cvtColor(eye, cv2.COLOR_BGR2BGRA)
#  マスク画像をもとに、白色部分を透明化
# eye_dst[eye_mask, 3] = 0

# faceの両目を識別(識別器の読み込み)
eye_path = "./haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(eye_path)
eyes = eye_cascade.detectMultiScale(face)
# print(eyes)
# [[343 142  28  28] ・・・くちが識別される罠
#  [365  94  29  29]
#  [326 103  25  25]]

# faceの左目の位置
x_l = eyes[1][0]
y_l = eyes[1][1]
# faceの左目の大きさ
w_l = eyes[1][2]
h_l = eyes[1][3]
# faceの右目の位置
x_r = eyes[2][0]
y_r = eyes[2][1]
# faceの右目の大きさ
w_r = eyes[2][2]
h_r = eyes[2][3]  #
# eyeをfaceの左目の大きさにあわせる
resized_eye1 = cv2.resize(clear_eye, dsize=(w_l, h_l))
# eyeをfaceの左目の大きさにあわせる
resized_eye2 = cv2.resize(clear_eye, dsize=(w_r, h_r))
# print(resized_eye.shape)
# 28,28,3 確認

# 座標を指定してfaceとeyeの画像を入れ替える
face[y_l : h_l + y_l, x_l : w_l + x_l] = resized_eye1
face[y_r : h_r + y_r, x_r : w_r + x_r] = resized_eye2
cv2.imwrite("images/paste.jpg", face)  #
