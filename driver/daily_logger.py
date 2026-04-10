# -*- coding: utf-8 -*-
import csv
import os
from datetime import datetime


class DailyLogger:
    """每日日志记录器 - 自动按日期创建日志文件并记录数据"""

    def __init__(self, log_dir="logs", file_extension="csv", delimiter=","):
        """
        初始化日志记录器

        参数:
            log_dir (str): 日志文件存放目录
            file_extension (str): 日志文件扩展名，支持 'csv' 或 'txt'
            delimiter (str): CSV文件分隔符
        """
        self.log_dir = log_dir
        self.file_extension = file_extension.lower()
        self.delimiter = delimiter
        self.headers = None

        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 初始化今日日期和文件路径
        self._update_today_file()

    def _update_today_file(self):
        """更新今日日期和文件路径"""
        today = datetime.now().strftime("%Y-%m-%d")
        self.today = today
        self.current_file = os.path.join(self.log_dir, f"{today}.{self.file_extension}")

        # 检查文件是否存在，确定是否需要写入标题行
        self.file_exists = os.path.exists(self.current_file)

    def set_headers(self, headers):
        """设置CSV文件的标题行"""
        self.headers = headers

    def log(self, data):
        """
        记录数据到今日日志文件

        参数:
            data (list/dict): 要记录的数据，可以是列表或字典
        """
        # 检查是否需要更新日期（跨天）
        self._update_today_file()

        try:
            # 写入CSV格式
            if self.file_extension == "csv":
                self._log_to_csv(data)
            # 写入纯文本格式
            else:
                self._log_to_text(data)

        except Exception as e:
            print(f"记录日志时出错: {e}")

    def _log_to_csv(self, data):
        """以CSV格式写入数据"""
        # 确保有标题行
        if self.headers is None:
            raise ValueError("使用CSV格式时必须先设置标题行 (set_headers)")

        # 处理字典格式的数据
        if isinstance(data, dict):
            # 将字典转换为列表（按标题行顺序）
            data_row = [data.get(header, "") for header in self.headers]
        else:
            data_row = data

        with open(self.current_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.delimiter)

            # 如果文件是新建的，写入标题行
            if not self.file_exists:
                writer.writerow(self.headers)
                self.file_exists = True

            # 写入数据行
            writer.writerow(data_row)

    def _log_to_text(self, data):
        """以纯文本格式写入数据"""
        # 将数据转换为字符串
        if isinstance(data, list) or isinstance(data, tuple):
            data_str = self.delimiter.join(map(str, data))
        elif isinstance(data, dict):
            data_str = self.delimiter.join(f"{k}={v}" for k, v in data.items())
        else:
            data_str = str(data)

        # 追加到文件
        with open(self.current_file, 'a', encoding='utf-8') as f:
            f.write(f"{data_str}\n")


# 使用示例
if __name__ == "__main__":
    # 创建CSV格式的日志记录器
    logger = DailyLogger(log_dir="operation_logs", file_extension="csv")
    logger.set_headers(["时间", "操作", "状态", "详情"])

    # 记录操作结果
    logger.log({
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "操作": "数据导入",
        "状态": "成功",
        "详情": "导入了100条记录"
    })

    # 记录另一条操作结果
    logger.log([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "系统检查",
        "警告",
        "磁盘空间不足"
    ])

    # 创建纯文本格式的日志记录器
    text_logger = DailyLogger(log_dir="system_logs", file_extension="txt", delimiter=" | ")
    text_logger.log(f"系统启动成功 - {datetime.now()}")