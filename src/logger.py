"""记录整个过程会发生的事 包括执行过程
    具体操作还是查文档"""

import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(filename=LOG_FILE_PATH,
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.info())

