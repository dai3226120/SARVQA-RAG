import pandas as pd
import chardet

def convert_csv_to_utf8(input_file_path, output_file_path=None):
    """
    检测CSV文件编码并转换为UTF-8
    
    Args:
        input_file_path (str): 输入文件路径
        output_file_path (str, optional): 输出文件路径。如果为None，则在原文件名后加'_utf8'
    """
    
    # 1. 检测文件编码
    print("🔍 正在检测文件编码...")
    with open(input_file_path, 'rb') as f:
        raw_data = f.read(100000)  # 读取前100KB进行检测，通常足够准确
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        print(f"📊 检测结果: 编码={encoding}, 置信度={confidence:.2f}")
    
    # 2. 读取数据
    try:
        # 尝试读取CSV
        df = pd.read_csv(input_file_path, encoding=encoding)
        print("✅ 数据读取成功！")
    except UnicodeDecodeError as e:
        print(f"❌ 读取失败: {e}")
        print("尝试使用错误忽略参数读取...")
        # 如果严格读取失败，尝试强制读取（忽略错误字符）
        df = pd.read_csv(input_file_path, encoding=encoding, on_bad_lines='skip')
        print("⚠️ 数据已读取，但部分行可能丢失。")
    
    # 3. 设置输出路径
    if output_file_path is None:
        # 默认输出路径：原文件名 + '_utf8.csv'
        if input_file_path.lower().endswith('.csv'):
            output_file_path = input_file_path[:-4] + '_utf8.csv'
        else:
            output_file_path = input_file_path + '_utf8.csv'
    
    # 4. 保存为UTF-8
    print(f"💾 正在将文件保存为 UTF-8 编码: {output_file_path}")
    df.to_csv(output_file_path, index=False, encoding='utf-8-sig') # utf-8-sig 可以防止Excel打开乱码
    
    print("🎉 转换完成！")

# --- 主程序 ---
if __name__ == "__main__":
    # 请修改这里的文件名为你实际的文件名
    # input_filename = "./dataset_analysis/origin_dataset/Landsat30-AU-VQA-train.csv" 
    # input_filename = "./dataset_analysis/origin_dataset/2024EarthVQA_QA.csv" 
    input_filename = "./dataset_analysis/origin_dataset/SARLANG-1M_all.csv"
    
    # 执行转换
    convert_csv_to_utf8(input_filename)