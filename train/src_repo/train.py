from ultralytics import YOLO
from pathlib import Path
from config import load_config 
cfg = load_config() 


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    data_yaml = repo_root / "dataset" / "data.yaml"
    model = YOLO(cfg["pre_model"])
    results = model.train(
        data=data_yaml,
        time=cfg["time"],
        #epochs=10,           # 训练轮数
        batch=cfg["batch"],          # 批处理大小
        imgsz=cfg["imgsz"],            # 输入图像大小 320 640 960 1280
        device=0,             # 使用GPU设备
        workers=cfg["workers"],      # 数据加载器的工作线程数
        patience=cfg["patience"],          # 早停的耐心值
        project=cfg["project"],        # 结果保存的项目目录
        name=cfg["name"],           # 结果保存的子目录名称
        exist_ok=True,        
        cache=cfg["cache"],          # 是否使用缓存
        pretrained=True,   
        plots=False,         # 是否生成训练过程的可视化图表
        save_json=False,     # 是否保存训练结果为JSON文件
        amp=True            
    )
    
