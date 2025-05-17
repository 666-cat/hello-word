# hello-word
此存储库用于联系GitHub流
输出数据
def output_data(df, output_type='print'):
    if output_type == 'print':
        # 打印前5行和数据基本信息
        print("\n数据预览（前5行）:")
        print(df.head())
        
        print("\n数据基本信息:")
        print(df.info())
        
        print("\n数据统计摘要:")
        print(df.describe())
    
    elif output_type == 'csv':
        # 保存为 CSV 文件
        output_path = Path.cwd() / "cleaned_data.csv"
        df.to_csv(output_path, index=False)
        print(f"数据已保存为 CSV 文件: {output_path}")
    
    elif output_type == 'excel':
        # 保存为 Excel 文件
        output_path = Path.cwd() / "cleaned_data.xlsx"
        df.to_excel(output_path, index=False)
        print(f"数据已保存为 Excel 文件: {output_path}")
