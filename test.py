import os
from dotenv import load_dotenv

# .envファイルの内容を読み込みます
load_dotenv()

# os.environを用いて環境変数を表示させます
print(os.environ['Consumer_key'])