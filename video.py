import cv2
import os

def images_to_video(input_dir, output_path, fps=5):
    # 获取输入目录中的所有图像文件
    images = [img for img in os.listdir(input_dir) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # 确保按文件名顺序读入

    if not images:
        print("No images found in the directory.")
        return

    # 读取第一张图片以获取尺寸信息
    first_image_path = os.path.join(input_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 遍历所有图片并写入视频
    for image in images:
        image_path = os.path.join(input_dir, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放 VideoWriter 对象
    video.release()
    print(f"The video has been saved as {output_path}")

# 使用示例
input_directory = '/vePFS001/luhao/Code/CLTGM/exp/21'  # 替换为你的图片文件夹路径
output_video_path = 'output_video.mp4'  # 输出视频文件名
images_to_video(input_directory, output_video_path)
