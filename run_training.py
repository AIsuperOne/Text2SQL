from modules.training_manager import BatchTrainer
import os

def main():
    print("开始训练指标定义...")
    trainer = BatchTrainer()
    
    # 定义指标文件路径
    metrics_yaml_path = r"C:\Users\Administrator\PYMo\SuperMO\Text2SQL\all_metrics.yaml"
    
    if not os.path.exists(metrics_yaml_path):
        print(f"错误: 指标文件不存在于 {metrics_yaml_path}")
        return
        
    count = trainer.train_from_metrics_yaml(metrics_yaml_path)
    print(f"训练完成，共处理了 {count} 个新文档。")

if __name__ == "__main__":
    main()
