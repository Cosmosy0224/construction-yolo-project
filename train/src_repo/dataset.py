import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from glob import glob
from sklearn.model_selection import train_test_split
from config import load_config
cfg = load_config()


class_names = cfg["class_names"]

data_root = Path("/home/data")
class_to_id = {name: idx for idx, name in enumerate(class_names)}

folders = sorted(str(p) for p in data_root.iterdir() if p.is_dir())
print("找到以下文件夹:") 
for folder in folders:
    print(f"  {folder}")

jpg_count = 0
bz_count = 0
output_dir = "/project/train/dataset"
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
train_images_dir = os.path.join(images_dir, "train")
val_images_dir = os.path.join(images_dir, "val")
train_labels_dir = os.path.join(labels_dir, "train")
val_labels_dir = os.path.join(labels_dir, "val")

for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
    os.makedirs(dir_path, exist_ok=True)

def convert_voc_to_yolo(xml_path):
    global bz_count
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size=root.find('size')
    if size is None:
        raise ValueError(f"XML中缺少size节点: {xml_path}")
    try:
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)
    except (ValueError, TypeError) as e:
        raise ValueError(f"无法解析图片尺寸: {xml_path}") from e

    yolo_annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in class_to_id:
            #计数
            bz_count += 1

            #print(f"类别 '{name}' 不在类别列表中，跳过")
            continue
        class_id = class_to_id[name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


def process_folder(folder_path, all_data):
    from PIL import Image, UnidentifiedImageError
    xml_files = glob(os.path.join(folder_path, "*.xml"))
    
    for xml_path in xml_files:
        base_name = os.path.splitext(xml_path)[0]
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = base_name + ext
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if img_path is None:
            print(f"找不到 {xml_path} 对应的图片文件")
            continue
        try:
            with Image.open(img_path) as im:
                im.verify()
        except (UnidentifiedImageError, OSError, IOError, SyntaxError) as e:
            print(f"损坏图片 {img_path}，已删除并跳过: {e}")
            try:
                os.remove(img_path)
            except Exception:
                pass
            try:
                os.remove(xml_path)
            except Exception:
                pass
            continue
        all_data.append((img_path, xml_path))

def copy_and_convert(item, dest_images_dir, dest_labels_dir):
    global jpg_count, bz_count
    img_path, xml_path = item
    img_filename = os.path.basename(img_path)
    dest_img_path = os.path.join(dest_images_dir, img_filename)
    label_filename = os.path.splitext(img_filename)[0] + '.txt'
    dest_label_path = os.path.join(dest_labels_dir, label_filename)
    try:
        yolo_annotations = convert_voc_to_yolo(xml_path)
        if not yolo_annotations:
            jpg_count += 1
            #print(f"图片 {img_path} 没有可训练的标注，已跳过")
            return False

        shutil.copy2(img_path, dest_img_path)

        with open(dest_label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        return True
    except Exception as e:
        print(f"处理 {img_path} 时出错: {e}")
        return False

def create_dataset_yaml():
    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

def main():
    all_data = []
    for folder in folders:
        process_folder(folder, all_data)
    print(f"总共找到 {len(all_data)} 个图片-标注对")
    
    if len(all_data) == 0:
        print("没有找到任何数据")
        return
    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
    
    print(f"训练集数量: {len(train_data)}")
    print(f"验证集数量: {len(val_data)}")
    
    print("正在处理训练集...")
    success_count = 0
    for idx, item in enumerate(train_data):
        if copy_and_convert(item, train_images_dir, train_labels_dir):
            success_count += 1
    print(f"训练集成功处理: {success_count}/{len(train_data)}")

    print("正在处理验证集...")
    success_count = 0
    for idx, item in enumerate(val_data):
        if copy_and_convert(item, val_images_dir, val_labels_dir):
            success_count += 1
    print(f"验证集成功处理: {success_count}/{len(val_data)}")
    print(f"总共跳过 {jpg_count} 张没有标注的图片")
    print(f"总共跳过 {bz_count} 个不在类别列表中的标注对象")
    create_dataset_yaml()

if __name__ == "__main__":
    main()
