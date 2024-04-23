import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download


# 下载单个文件
def download_model(source_url):
    if 'blob' in source_url:
        sp = '/blob/main/'
    else:
        sp = '/resolve/main/'

    if 'huggingface.co' in source_url:
        url = 'https://huggingface.co/'
    else:
        url = 'https://hf-mirror.com'

    location = source_url.split(sp)
    repo_id = location[0].strip(url)  # 仓库ID，例如："BlinkDL/rwkv-4-world"

    local_dir = r'C:\Users\26522\Desktop\code'
    cache_dir = local_dir + "/cache"
    filename = location[1]  # 大模型文件，例如："RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth"

    print(
        f'开始下载\n仓库：{repo_id}\n大模型：{filename}\n如超时不用管，会自定继续下载，直至完成。中途中断，再次运行将继续下载。')
    while True:
        try:
            hf_hub_download(cache_dir=cache_dir,
                            local_dir=local_dir,
                            repo_id=repo_id,
                            filename=filename,
                            local_dir_use_symlinks=False,
                            resume_download=True,
                            etag_timeout=100
                            )
        except Exception as e:
            print(e)
        else:
            print(f'下载完成，大模型保存在：{local_dir}\{filename}')
            break


if __name__ == '__main__':
    # # 使用镜像下载单个文件
    # source_url = "https://hf-mirror.com/wangqixun/YamerMIX_v8/tree/main"
    # download_model(source_url)

    local_dir = r'./code'
    cache_dir = local_dir + "/cache"
    repo_id = "yzd-v/DWPose"
    filename = "dw-ll_ucoco_384.onnx"
    # 下载单个文件
    hf_hub_download(cache_dir=cache_dir,            # 缓存位置
                    local_dir=local_dir,            # 下载后存储位置
                    repo_id=repo_id,                # 项目库名
                    filename=filename,              # 下载文件名
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    etag_timeout=100
                    )
    # 下载整个项目
    # snapshot_download(repo_id="wangqixun/YamerMIX_v8", local_dir="./base")
