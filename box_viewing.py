import pickle
import cv2
import numpy as np
import PIL.Image as Image
# import matplotlib
# matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

res_path="./DATA/hico_det_test.pkl"
sample_list=[1]
len_List=[2]
res = pickle.load(open(res_path, "rb"))

width=512
height=512
k=0
for index in sample_list:
    gray_image = np.full((height, width, 3), 128, dtype=np.uint8)
    res_sample=res[index-1]
    subject_box=res_sample["subject_boxes"]
    object_box=res_sample["object_boxes"]

    subject_len=len(subject_box)
    for i in range(len_List[k]):
        x_min,y_min,x_max,y_max=subject_box[i]
        x_min=int(x_min*width)
        y_min=int(y_min*height)
        x_max=int(x_max*width)
        y_max=int(y_max*height)

        top_left=(x_min,y_min)
        bottom_right=(x_max,y_max)
        cv2.rectangle(gray_image,top_left,bottom_right,[255,0,0],3)

        label = res_sample["subject_phrases"][i]  # 文本内容
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体样式
        font_scale = 1  # 字体大小缩放因子
        font_color = (255, 0, 0)  # 白色文字
        text_position = (top_left[0], top_left[1] - 10)  # 文本位置偏移
        cv2.putText(gray_image, label, text_position, font, font_scale, font_color, 2)


        x_min, y_min, x_max, y_max = object_box[i]
        x_min = int(x_min * width)
        y_min = int(y_min * height)
        x_max = int(x_max * width)
        y_max = int(y_max * height)

        top_left = (x_min, y_min)
        bottom_right = (x_max, y_max)
        cv2.rectangle(gray_image, top_left, bottom_right, [0, 0, 255], 3)

        label = res_sample["object_phrases"][i]  # 文本内容
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体样式
        font_scale = 1  # 字体大小缩放因子
        font_color = (0, 0, 255)  # 白色文字
        text_position = (top_left[0], top_left[1] - 10)  # 文本位置偏移
        cv2.putText(gray_image, label, text_position, font, font_scale, font_color, 2)


        gray_image=cv2.cvtColor(gray_image,cv2.COLOR_BGR2RGB)
    k+=1
    image_pil=Image.fromarray(gray_image)
    image_pil.save(f"./inference_input/input_{index}.jpg")


        # gray_image.show()
        #
        # plt.axis("off")
        # plt.show()


print(0)