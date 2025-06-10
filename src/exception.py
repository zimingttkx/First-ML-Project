import logging
import sys

# 配置日志（这是一个好习惯，可以将日志输出到文件或控制台）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""Python 自定义异常检测 具体实现可以查看 python异常处理说明文档"""

def error_message_detail(error, error_detail: sys):
    """
    获取异常的详细信息。
    Args:
        error (Exception): 捕获到的异常对象。
        error_detail (sys module): sys 模块，用于获取异常追踪信息。
    Returns:
        str: 包含文件名、行号和错误消息的详细错误字符串。
    """
    # exc_info() 返回 (type, value, traceback)
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = str(error) # 直接获取异常对象的字符串表示

    # 格式化输出错误信息
    return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, error_message
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        自定义异常类。
        Args:
            error_message (Exception or str): 原始的异常对象或者一个错误消息字符串。
            error_detail (sys module): sys 模块，用于获取异常追踪信息。
        """
        # 调用父类Exception的构造函数，通常传入一个消息字符串
        # 这里的 error_message 如果是 Exception 对象，super() 会尝试调用它的 __str__
        # 或者我们可以在这里传入一个通用的、预设的消息，然后将详细信息在 self.error_message 中处理
        super().__init__("An error occurred in the application.") # 先给一个通用的父类消息

        # 调用 error_message_detail 来生成详细的错误信息
        # 确保传入给 error_message_detail 的第一个参数是实际的异常对象
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        """
        定义异常的字符串表示，当打印异常或将其转换为字符串时会被调用。
        """
        return self.error_message


