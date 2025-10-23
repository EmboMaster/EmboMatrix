import cv2
import os

def create_video_from_images(image_folder, video_filename, start,fps=30):
    # 获取文件夹中所有以 'xx_room' 开头的图片文件
    image_files = [f for f in os.listdir(image_folder) if f.startswith(start) and f.endswith('.jpg')]
# 确保至少有一张图片
    if len(image_files) == 0:
        print("没有找到图片！")
        return

    # 按文件修改时间排序
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(image_folder, x)))

    # 读取第一张图片，获取其尺寸
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择视频编码格式
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    # 遍历所有图片并将其添加到视频中
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)

        if img is not None:
            out.write(img)  # 将图片写入视频

    # 释放视频写入对象
    out.release()
    print(f"视频已成功保存为 {video_filename}")

image_folder = 'omnigibson/examples/scenes/picture_of_room/Beechwood_0_garden_Full'  # 图片文件夹路径
video_filename ='omnigibson/examples/scenes/picture_of_room/livingroom_video_20fps.mp4'  # 输出视频文件名
start = 'living_room'
create_video_from_images(image_folder, video_filename, start,fps=20)