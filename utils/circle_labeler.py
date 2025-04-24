import os
import cv2
import json
import numpy as np
import re
import shutil
import tkinter as tk
from tkinter import filedialog
import time

class CircleLabeler:
    def __init__(self, image_dir=None, output_file=None):
        # 使用GUI选择文件夹和输出文件
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        # 选择图片文件夹
        if image_dir is None:
            image_dir = filedialog.askdirectory(title="选择包含图片的文件夹")
            if not image_dir:
                print("未选择文件夹，程序退出")
                exit()
        self.image_dir = image_dir
        
        # 选择输出文件
        if output_file is None:
            default_output = os.path.join(image_dir, "circle_centers.json")
            output_file = filedialog.asksaveasfilename(
                title="保存标签文件",
                initialdir=image_dir,
                initialfile="circle_centers.json",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            if not output_file:
                output_file = default_output
        self.output_file = output_file
        
        # 创建删除图片的目录
        self.trash_dir = os.path.join(image_dir, "_deleted_images")
        os.makedirs(self.trash_dir, exist_ok=True)
            
        # 状态变量
        self.images = []
        self.current_idx = 0
        self.labels = {}
        self.window_name = "圆心标定工具"
        self.current_image = None  # 存储当前图像
        
        # 加载图片和标签
        self.load_images()
        self.load_labels()
        
        # 初始化UI
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
    
    def extract_number(self, filename):
        """从文件名中提取数字，用于数字排序"""
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    def load_images(self):
        """加载文件夹中的所有图片并按数字顺序排序"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        self.images = [f for f in os.listdir(self.image_dir) 
                     if os.path.splitext(f.lower())[1] in valid_extensions]
        
        # 按照文件名中的数字进行排序
        try:
            self.images.sort(key=self.extract_number)
        except:
            # 如果数字排序失败，使用普通排序
            self.images.sort()
            
        print(f"找到 {len(self.images)} 张图片")
    
    def load_labels(self):
        """加载已有标签"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    self.labels = json.load(f)
                print(f"已加载 {len(self.labels)} 个标签")
            except Exception as e:
                print(f"加载标签出错: {e}")
                self.labels = {}
    
    def save_labels(self):
        """保存标签"""
        with open(self.output_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"已保存 {len(self.labels)} 个标签到 {self.output_file}")
    
    def next_image(self):
        """切换到下一张图片"""
        if self.images:
            self.current_idx = (self.current_idx + 1) % len(self.images)
            self.display_current()
    
    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标事件"""
        if len(self.images) == 0 or self.current_image is None:
            return
            
        # 当前图像
        img_name = self.images[self.current_idx]
        
        if event == cv2.EVENT_LBUTTONDOWN:  # 标记圆心
            self.labels[img_name] = [x, y]
            print(f"标记 {img_name}: ({x}, {y})")
            
            # 立即在点击位置显示标识点
            display_img = self.current_image.copy()
            cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)  # 圆心(实心点)
            cv2.circle(display_img, (x, y), 20, (0, 255, 0), 2)  # 圆环
            
            # 显示状态信息
            progress = f"[{self.current_idx+1}/{len(self.images)}]"
            file_info = f"{img_name}"
            status = f"{progress} {file_info} - 已标记: ({x}, {y})"
            cv2.putText(display_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 显示图像
            cv2.imshow(self.window_name, display_img)
            cv2.waitKey(300)  # 短暂显示标记点
            
            # 保存标签并切换到下一张
            self.save_labels()
            self.next_image()
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # 取消标记
            if img_name in self.labels:
                del self.labels[img_name]
                print(f"取消标记 {img_name}")
                self.save_labels()
                self.display_current()
    
    def delete_current_image(self):
        """删除当前图片(移动到回收站)"""
        if not self.images:
            print("没有图片可删除")
            return
            
        img_name = self.images[self.current_idx]
        src_path = os.path.join(self.image_dir, img_name)
        dst_path = os.path.join(self.trash_dir, img_name)
        
        try:
            # 移动到回收站
            shutil.move(src_path, dst_path)
            print(f"已移动 {img_name} 到回收站")
            
            # 从标签中删除(如果存在)
            if img_name in self.labels:
                del self.labels[img_name]
                self.save_labels()
            
            # 从列表中删除
            self.images.pop(self.current_idx)
            
            # 更新索引
            if self.images:
                self.current_idx = min(self.current_idx, len(self.images) - 1)
                self.display_current()
            else:
                print("没有更多图片")
                self.display_empty()
                
        except Exception as e:
            print(f"删除图片失败: {e}")
    
    def display_current(self):
        """显示当前图片及标记"""
        if not self.images:
            self.display_empty()
            return
            
        # 读取当前图片
        img_name = self.images[self.current_idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"无法读取图片: {img_path}")
                self.display_error()
                return
                
            # 保存当前图像
            self.current_image = image.copy()
            
            # 制作显示用的副本
            display_img = image.copy()
            
            # 显示标记
            if img_name in self.labels:
                cx, cy = self.labels[img_name]
                cv2.circle(display_img, (cx, cy), 5, (0, 255, 0), -1)  # 圆心(实心点)
                cv2.circle(display_img, (cx, cy), 20, (0, 255, 0), 2)  # 圆环
            
            # 状态信息
            progress = f"[{self.current_idx+1}/{len(self.images)}]"
            file_info = f"{img_name}"
            status = f"{progress} {file_info}"
            if img_name in self.labels:
                status += f" - 已标记: ({self.labels[img_name][0]}, {self.labels[img_name][1]})"
            
            # 添加信息
            cv2.putText(display_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 显示图像
            cv2.imshow(self.window_name, display_img)
            
        except Exception as e:
            print(f"显示图片出错: {e}")
            self.display_error()
    
    def display_empty(self):
        """显示空状态"""
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(canvas, "没有图片", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "按 'q' 退出", (320, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.imshow(self.window_name, canvas)
    
    def display_error(self):
        """显示错误状态"""
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(canvas, "图片加载错误", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(canvas, "按 'd' 删除此图片, 或 'n' 继续", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.imshow(self.window_name, canvas)
    
    def run(self):
        """运行标注工具"""
        if not self.images:
            print("没有找到图片")
            self.display_empty()
        else:
            self.display_current()
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # q或ESC退出
                break
                
            elif key == ord('n') or key == 83:  # n或→下一张
                self.next_image()
                    
            elif key == ord('p') or key == 81:  # p或←上一张
                if self.images:
                    self.current_idx = (self.current_idx - 1) % len(self.images)
                    self.display_current()
                    
            elif key == ord('d'):  # 删除当前图片
                self.delete_current_image()
        
        # 退出前保存
        self.save_labels()
        cv2.destroyAllWindows()
        print("程序结束")

# 直接运行
if __name__ == "__main__":
    try:
        labeler = CircleLabeler()
        labeler.run()
    except Exception as e:
        print(f"程序运行出错: {e}")