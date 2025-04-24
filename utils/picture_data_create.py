import cv2
import numpy as np
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir):
    """
    从视频文件提取帧并保存到输出目录
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出目录路径
    
    返回:
        提取的帧数
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频属性
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"处理视频: {video_path}")
    print(f"总帧数: {frame_count}")
    print(f"帧率: {fps}")
    
    # 提取帧
    frame_idx = 0
    success = True
    
    while success:
        success, frame = video.read()
        if success:
            output_path = os.path.join(output_dir, f"p{frame_idx+1}.png")
            cv2.imwrite(output_path, frame)
            frame_idx += 1
    
    video.release()
    print(f"已提取 {frame_idx} 帧到 {output_dir}")
    
    return frame_idx

def preprocess_image(image, target_size=(256, 256), normalize=True, normalize_range=(-1, 1)):
    """
    预处理图像：转灰度、调整大小、归一化
    
    参数:
        image: 输入图像
        target_size: 目标图像大小，默认(256, 256)
        normalize: 是否归一化像素值
        normalize_range: 归一化范围，默认为(-1, 1)
        
    返回:
        预处理后的图像
    """
    # 转换为灰度图
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 调整为固定大小
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 归一化像素值
    if normalize:
        # 归一化到[0,1]
        normalized = resized.astype(np.float32) / 255.0
        
        # 如果需要归一化到[-1,1]
        if normalize_range == (-1, 1):
            normalized = normalized * 2 - 1
    else:
        normalized = resized
    
    return normalized

def augment_images(input_dir, output_dir, augmentation_per_image=5):
    """
    对输入目录中的图像进行数据增强并保存到输出目录
    
    参数:
        input_dir: 输入图像目录
        output_dir: 增强后的图像输出目录
        augmentation_per_image: 每张图像生成的增强图像数量
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"开始对 {len(image_files)} 张图像进行数据增强...")
    
    img_counter = 0
    
    # 处理每张图像
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像: {img_path}")
            continue
        
        # 先保存原始图像
        img_counter += 1
        output_path = os.path.join(output_dir, f"p{img_counter}.png")
        cv2.imwrite(output_path, img)
        
        # 对每张图像生成多个增强版本
        for i in range(augmentation_per_image):
            # 应用随机增强
            augmented = apply_random_augmentations(img)
            
            # 保存增强后的图像
            img_counter += 1
            output_path = os.path.join(output_dir, f"p{img_counter}.png")
            cv2.imwrite(output_path, augmented)
    
    print(f"已生成 {img_counter} 张增强图像到 {output_dir}")

def position_circle(image):
    """
    将干涉圆环放置在图像的不同位置，确保数据集中的圆环分布在各个区域
    
    参数:
        image: 输入图像
        
    返回:
        修改后的图像，圆环位于随机位置
    """
    h, w = image.shape[:2]
    
    # 估计原始圆环的位置和大小
    # 假设原始圆环在图像中心
    center_x_orig, center_y_orig = w // 2, h // 2
    radius = min(w, h) // 3  # 假设这是圆环的近似半径
    
    # 定义图像的9个区域（3x3网格）
    grid_w, grid_h = w // 3, h // 3
    
    # 随机选择一个区域作为新的圆心位置
    # 为了让圆环完全可见，我们将位置限制在一定范围内
    grid_x = random.randint(0, 2)
    grid_y = random.randint(0, 2)
    
    # 计算新的中心位置，在选定区域内随机
    padding = radius // 2  # 确保圆环不会太靠近边缘
    new_center_x = random.randint(grid_x * grid_w + padding, (grid_x + 1) * grid_w - padding)
    new_center_y = random.randint(grid_y * grid_h + padding, (grid_y + 1) * grid_h - padding)
    
    # 计算位移量
    dx = new_center_x - center_x_orig
    dy = new_center_y - center_y_orig
    
    # 应用平移
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    result = cv2.warpAffine(image, translation_matrix, (w, h), 
                          flags=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REFLECT)
    
    return result

def apply_random_augmentations(image):
    """
    对图像应用随机增强
    
    参数:
        image: 输入图像
        
    返回:
        增强后的图像
    """
    # 复制原始图像
    result = image.copy()
    
    # 每次增强都先应用位置变化，确保圆环分布在各个位置
    if random.random() < 0.7:  # 70%的概率应用位置变化
        result = position_circle(result)
    
    # 随机决定应用哪些增强
    augmentations = [
        rotate_image,
        translate_image,
        scale_image,
        adjust_brightness,
        adjust_contrast,
        apply_gamma_correction,
        add_gaussian_noise,
        apply_gaussian_blur,
        add_random_occlusion,
        change_background,
        simulate_exposure
    ]
    
    # 随机选择1-3种增强方法
    num_augmentations = random.randint(1, 3)
    selected_augmentations = random.sample(augmentations, num_augmentations)
    
    # 应用选择的增强
    for augmentation in selected_augmentations:
        result = augmentation(result)
    
    return result

# ===== 几何变换函数 =====

def rotate_image(image):
    """随机旋转图像"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # 随机旋转角度 (0-360度)
    angle = random.uniform(0, 360)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h), 
                          flags=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REFLECT)

def translate_image(image):
    """随机平移图像"""
    h, w = image.shape[:2]
    # 最大平移为图像尺寸的20%
    max_tx = w * 0.2
    max_ty = h * 0.2
    
    # 随机平移量
    tx = random.uniform(-max_tx, max_tx)
    ty = random.uniform(-max_ty, max_ty)
    
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, translation_matrix, (w, h), 
                          flags=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REFLECT)

def scale_image(image):
    """随机缩放图像"""
    h, w = image.shape[:2]
    # 随机缩放因子 (0.8-1.2)
    scale = random.uniform(0.8, 1.2)
    
    scaled_h, scaled_w = int(h * scale), int(w * scale)
    
    # 缩放图像
    scaled_img = cv2.resize(image, (scaled_w, scaled_h), 
                            interpolation=cv2.INTER_LINEAR)
    
    # 创建原始大小的画布
    result = np.zeros_like(image)
    
    # 计算粘贴位置
    paste_x = max(0, (w - scaled_w) // 2)
    paste_y = max(0, (h - scaled_h) // 2)
    
    # 计算要粘贴的区域
    if scaled_w > w:
        src_x, dst_x = (scaled_w - w) // 2, 0
        width = w
    else:
        src_x, dst_x = 0, paste_x
        width = scaled_w
        
    if scaled_h > h:
        src_y, dst_y = (scaled_h - h) // 2, 0
        height = h
    else:
        src_y, dst_y = 0, paste_y
        height = scaled_h
    
    # 将缩放后的图像粘贴到画布上
    result[dst_y:dst_y+height, dst_x:dst_x+width] = scaled_img[src_y:src_y+height, src_x:src_x+width]
    
    return result

# ===== 光学特性变换函数 =====

def adjust_brightness(image):
    """调整图像亮度"""
    # 随机亮度因子 (0.8-1.2，±20%)
    factor = random.uniform(0.8, 1.2)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image):
    """调整图像对比度"""
    # 随机对比度因子 (0.8-1.2，±20%)
    factor = random.uniform(0.8, 1.2)
    
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    result = (image - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_gamma_correction(image):
    """应用伽马校正"""
    # 随机伽马值 (0.8-1.2)
    gamma = random.uniform(0.8, 1.2)
    
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def add_gaussian_noise(image):
    """添加高斯噪声"""
    # 随机噪声标准差 (5-15)
    sigma = random.uniform(5, 15)
    
    noise = np.random.normal(0, sigma, image.shape).astype(np.int32)
    noisy_img = image.astype(np.int32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_gaussian_blur(image):
    """应用高斯模糊"""
    # 随机模糊标准差 (0.5-1.5)
    sigma = random.uniform(0.5, 1.5)
    
    kernel_size = int(2 * round(3 * sigma) + 1)  # 确保核大小为奇数
    kernel_size = max(1, kernel_size)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

# ===== 场景模拟函数 =====

def add_random_occlusion(image):
    """添加随机遮挡"""
    result = image.copy()
    h, w = image.shape[:2]
    
    # 生成随机遮挡形状（矩形或圆形）
    occlusion_type = random.choice(['rectangle', 'circle'])
    
    # 随机位置和大小
    center_x = random.randint(w // 4, 3 * w // 4)
    center_y = random.randint(h // 4, 3 * h // 4)
    
    max_size = min(w, h) // 6  # 遮挡的最大尺寸
    
    if occlusion_type == 'rectangle':
        width = random.randint(max_size // 2, max_size)
        height = random.randint(max_size // 2, max_size)
        top_left = (max(0, center_x - width // 2), max(0, center_y - height // 2))
        bottom_right = (min(w, center_x + width // 2), min(h, center_y + height // 2))
        
        # 遮挡的随机颜色
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        cv2.rectangle(result, top_left, bottom_right, color, -1)
    else:  # circle
        radius = random.randint(max_size // 4, max_size // 2)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(result, (center_x, center_y), radius, color, -1)
    
    return result

def change_background(image):
    """修改图像背景，保持圆形干涉图案不变"""
    h, w = image.shape[:2]
    result = image.copy()
    
    # 为圆创建一个掩码（假设圆在中心）
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 3  # 干涉图案的近似半径
    
    # 创建圆形掩码
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist_from_center <= radius
    
    # 生成新背景
    background_type = random.choice(['gradient', 'noise', 'solid'])
    
    if background_type == 'gradient':
        # 创建渐变背景
        background = np.zeros_like(image)
        direction = random.choice(['horizontal', 'vertical', 'radial'])
        
        if direction == 'horizontal':
            for x in range(w):
                val = int(255 * x / w)
                background[:, x] = (val, val, val)
        elif direction == 'vertical':
            for y in range(h):
                val = int(255 * y / h)
                background[y, :] = (val, val, val)
        else:  # radial
            for y in range(h):
                for x in range(w):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    val = int(255 * dist / (np.sqrt(w**2 + h**2) / 2))
                    val = min(255, val)
                    background[y, x] = (val, val, val)
    
    elif background_type == 'noise':
        # 创建噪声背景
        background = np.random.randint(0, 255, size=image.shape, dtype=np.uint8)
    
    else:  # solid
        # 创建纯色背景
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        background = np.full_like(image, color)
    
    # 保留原始图像中的圆形
    result[~mask] = background[~mask]
    
    return result

def simulate_exposure(image):
    """模拟过度/不足曝光"""
    result = image.copy().astype(np.float32)
    
    # 随机选择过度或不足曝光
    over = random.choice([True, False])
    
    if over:
        # 过度曝光 - 剪裁高值
        threshold = random.uniform(150, 230)
        result[result > threshold] = 255
    else:
        # 曝光不足 - 剪裁低值
        threshold = random.uniform(30, 100)
        result[result < threshold] = 0
    
    return np.clip(result, 0, 255).astype(np.uint8)

def split_dataset(dataset_dir, target_total=900, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    将数据集分割为训练、验证和测试集
    
    参数:
        dataset_dir: 包含所有图像的目录
        target_total: 目标图像总数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 确保比例和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio != 1.0:
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
    
    # 计算每个集合的图像数量
    train_count = int(target_total * train_ratio)
    val_count = int(target_total * val_ratio)
    test_count = target_total - train_count - val_count
    
    print(f"目标分割: 训练集={train_count}张, 验证集={val_count}张, 测试集={test_count}张")
    
    # 创建目标目录
    train_dir = os.path.join(os.path.dirname(dataset_dir), "train")
    val_dir = os.path.join(os.path.dirname(dataset_dir), "val")
    test_dir = os.path.join(os.path.dirname(dataset_dir), "test")
    
    for directory in [train_dir, val_dir, test_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # 清空目录
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    
    # 获取所有图像文件
    all_images = [f for f in os.listdir(dataset_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_images)  # 随机打乱
    
    # 限制使用的图像数量
    selected_images = all_images[:target_total] if len(all_images) > target_total else all_images
    
    if len(selected_images) < target_total:
        print(f"警告: 只有 {len(selected_images)} 张图像可用，少于目标的 {target_total} 张")
    
    # 划分数据集
    train_images = selected_images[:train_count]
    val_images = selected_images[train_count:train_count+val_count]
    test_images = selected_images[train_count+val_count:train_count+val_count+test_count]
    
    # 复制文件到对应目录并应用预处理
    def copy_and_preprocess(image_list, source_dir, target_dir):
        for i, img_file in enumerate(image_list):
            src_path = os.path.join(source_dir, img_file)
            # 设置目标文件名格式为p1.png, p2.png等
            dst_path = os.path.join(target_dir, f"p{i+1}.png")
            
            # 读取图像
            img = cv2.imread(src_path)
            if img is None:
                print(f"警告: 无法读取图像: {src_path}")
                continue
            
            # 预处理图像
            processed = preprocess_image(img)
            
            # 保存预处理后的图像
            # 注意：归一化后的图像需要乘以255转回uint8才能用imwrite保存
            if processed.dtype == np.float32 or processed.dtype == np.float64:
                if np.min(processed) < 0:  # 如果是[-1,1]范围
                    save_img = ((processed + 1) / 2 * 255).astype(np.uint8)
                else:  # 如果是[0,1]范围
                    save_img = (processed * 255).astype(np.uint8)
            else:
                save_img = processed
                
            cv2.imwrite(dst_path, save_img)
    
    print("正在处理训练集...")
    copy_and_preprocess(train_images, dataset_dir, train_dir)
    
    print("正在处理验证集...")
    copy_and_preprocess(val_images, dataset_dir, val_dir)
    
    print("正在处理测试集...")
    copy_and_preprocess(test_images, dataset_dir, test_dir)
    
    print(f"数据集分割完成: 训练集={len(train_images)}张, 验证集={len(val_images)}张, 测试集={len(test_images)}张")
    return train_dir, val_dir, test_dir

def process_video_to_dataset(video_path, output_base_dir='data', target_total=900, augmentation_per_frame=None):
    """
    处理视频，提取帧并创建增强数据集，然后分割为训练、验证和测试集
    
    参数:
        video_path: 视频文件路径
        output_base_dir: 输出基础目录
        target_total: 目标图像总数
        augmentation_per_frame: 每帧创建的增强图像数量，如果为None则自动计算
    """
    # 创建临时目录和输出目录
    tmp_dir = os.path.join(output_base_dir, 'tmp')
    aug_dir = os.path.join(output_base_dir, 'augmented')
    
    for directory in [tmp_dir, aug_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # 清空目录
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    
    print(f"从视频提取帧: {video_path}")
    
    # 提取帧到临时目录
    num_frames = extract_frames(video_path, tmp_dir)
    
    print(f"已提取 {num_frames} 帧。")
    
    # 计算每帧需要的增强数量以达到目标总数
    if augmentation_per_frame is None:
        # 计算每帧平均需要的增强图像数，但至少为1
        augmentation_per_frame = max(1, (target_total - num_frames) // num_frames)
    
    print(f"每帧将生成 {augmentation_per_frame} 个增强版本，以接近目标的 {target_total} 张图像")
    
    # 获取所有提取帧的文件列表
    frame_files = sorted([f for f in os.listdir(tmp_dir) if f.endswith(('.png', '.jpg'))])
    
    # 图像计数器
    img_counter = 0
    
    # 处理每一帧
    print("生成增强训练数据...")
    for frame_file in tqdm(frame_files):
        frame_path = os.path.join(tmp_dir, frame_file)
        
        # 读取帧
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"警告: 无法读取帧: {frame_path}")
            continue
        
        # 先保存原始帧
        img_counter += 1
        output_path = os.path.join(aug_dir, f"p{img_counter}.png")
        cv2.imwrite(output_path, frame)
        
        # 如果已经达到目标数量，则停止处理
        if img_counter >= target_total:
            break
            
        # 生成增强版本
        for i in range(augmentation_per_frame):
            # 如果已经达到目标数量，则停止处理
            if img_counter >= target_total:
                break
                
            # 应用随机增强
            augmented = apply_random_augmentations(frame)
            
            # 保存增强后的图像
            img_counter += 1
            output_path = os.path.join(aug_dir, f"p{img_counter}.png")
            cv2.imwrite(output_path, augmented)
    
    print(f"已生成 {img_counter} 张增强图像到 {aug_dir}")
    
    # 分割数据集
    print("分割数据集为训练、验证和测试集...")
    train_dir, val_dir, test_dir = split_dataset(aug_dir, target_total=min(target_total, img_counter))
    
    # 清理临时目录
    print("清理临时文件...")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # 可选：清理增强目录，因为已经分割到各个子集中
    # shutil.rmtree(aug_dir, ignore_errors=True)
    
    print(f"处理完成。数据集位于:")
    print(f"训练集: {train_dir}")
    print(f"验证集: {val_dir}")
    print(f"测试集: {test_dir}")

# 使用示例
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python picture_data_create.py <视频文件路径> [输出基础目录] [目标图像总数]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'data'
    target_total = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    
    process_video_to_dataset(video_path, output_dir, target_total)

