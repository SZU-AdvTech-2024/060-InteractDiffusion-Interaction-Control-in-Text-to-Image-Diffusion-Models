import cv2
import numpy as np
import pickle
import os


def draw_boxes_with_labels(image_height, image_width, boxes_info, save_path):
    """
    在给定尺寸的图像上绘制带有标签的多个方框，并将图像保存到指定路径

    参数:
    image_height (int): 图像的高度
    image_width (int): 图像的宽度
    boxes_info (list): 包含多个方框信息的列表，每个方框信息为一个列表，其元素依次是左上角坐标x、左上角坐标y、右下角坐标x、右下角坐标y（均为[0, 1]的小数形式）以及标签，例如 [[x1_min, y1_min, x1_max, y1_max, "box1"], [x2_min, y2_min, x2_max, y2_max, "box2"],...]
    save_path (str): 图像保存的路径（包含文件名和扩展名，例如"./result_image.jpg"）
    """
    # 创建灰色背景图像
    image = np.full((image_height, image_width, 3), 128, dtype=np.uint8)

    count=0

    # 遍历每个方框信息列表
    for box_info in boxes_info:
        x_min, y_min, x_max, y_max, label = box_info
        x_min_px = int(x_min * image_width)
        y_min_px = int(y_min * image_height)
        x_max_px = int(x_max * image_width)
        y_max_px = int(y_max * image_height)

        # 根据标签设置颜色，这里简单示例两种颜色交替（可按需修改更复杂的颜色分配逻辑）
        if count == 0:
            color = (255, 0, 0)  # 蓝色
        elif count == 1:
            color = (0, 0, 255)  # 红色
        else:
            color = (0, 255, 0)  # 绿色，其他标签默认用绿色，可自行调整

        # 绘制方框
        cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), color, 2)
        # 在方框左上角附近添加标签
        # cv2.putText(image, label, (x_min_px, y_min_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        count+=1
    # 保存图像到指定路径
    cv2.imwrite(save_path, image)
    print(f"图像已成功保存至 {save_path}")


res_path = "DATA/hico_det_test.pkl"
res = pickle.load(open(res_path, "rb"))

sample_id=96
max_instance_num=1
count=0

sample=res[sample_id]
prompt=sample["prompt"]

instance_num=min(max_instance_num,len(sample["subject_phrases"]))

prompt_list=prompt.split(",")

text_prompt=",".join([prompt_list[i] for i in range(instance_num)])


subject_phrases = [sample["subject_phrases"][i] for i in range(instance_num)]
object_phrases = [sample["object_phrases"][i] for i in range(instance_num)]
action_phrases = [sample["action_phrases"][i] for i in range(instance_num)]
subject_boxes = [sample["subject_boxes"][i] for i in range(instance_num)]
object_boxes = [sample["object_boxes"][i] for i in range(instance_num)]

# prompt = "a yellow American robin, brown Maltipoo dog, a gray British Shorthair in a stream, alongside with trees and rocks"

# normalized (xmin,ymin,xmax,ymax)
object_boxes.extend(subject_boxes)
object_phrases.extend(subject_phrases)



# 示例用法
image_height = 512
image_width = 512

boxes=[]
for i in range(len(object_boxes)):
    box=object_boxes[i]
    phrase=object_phrases[i]
    box.extend([phrase])
    boxes.append(box)
# boxes = [
#     [0.2, 0.3, 0.6, 0.7, "box1"],
#     [0.4, 0.1, 0.8, 0.5, "box2"],
#     [0.1, 0.6, 0.3, 0.8, "box3"]
# ]

if not os.path.exists(f"/data/xiaoliangqiu/project/InteractDiff/inference_input{sample_id}"):
    os.makedirs(f"/data/xiaoliangqiu/project/InteractDiff/inference_input{sample_id}")

save_path = f"/data/xiaoliangqiu/project/InteractDiff/inference_input{sample_id}/result_image.jpg"  # 可自行修改保存路径及文件名，比如"./my_images/multiple_boxes_result.jpg"
draw_boxes_with_labels(image_height, image_width, boxes, save_path)
