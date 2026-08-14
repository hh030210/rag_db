# -*- coding: utf-8 -*-
"""
pack_and_upload.py
==================
把 search_service_deploy/ 整个目录打成 tar.gz，
然后通过 scp 上传到 81.70.191.196:/root/mingqiang/search_service_deploy.tar.gz

Windows 上建议使用 Paramiko 直接走 SSH。
先在本地 pip install paramiko。
"""

import os
import sys
import tarfile
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent  # search_service_deploy/
TAR_NAME = "search_service_deploy.tar.gz"


def make_tar_gz(src_dir: Path, dst_path: Path):
    print(f"[pack] 打包 {src_dir} -> {dst_path} ...")
    # Windows 用 mode='w:gz'，但可写性需要写二进制
    with tarfile.open(dst_path, "w:gz") as tar:
        # 进入 src_dir，把整个目录加进归档
        tar.add(src_dir, arcname=src_dir.name, filter=_filter)
    size_mb = dst_path.stat().st_size / 1024 / 1024
    print(f"[pack] [OK] 完成，{size_mb:.2f} MB")


def _filter(tarinfo):
    """跳过 __pycache__、.pyc、smoke 文件"""
    name = tarinfo.name
    if "__pycache__" in name:
        return None
    if name.endswith(".pyc"):
        return None
    if name.endswith("_smoke.py"):
        return None
    return tarinfo


def upload_ssh(tar_path: Path, host: str, port: int, user: str, password: str, remote_dir: str):
    """用 paramiko SSH 上传 + 解压"""
    try:
        import paramiko
    except ImportError:
        print("[ERROR] 请先 pip install paramiko")
        sys.exit(1)

    print(f"[ssh] 连接到 {user}@{host}:{port}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username=user, password=password, timeout=30)

    sftp = ssh.open_sftp()
    remote_tar = f"{remote_dir}/{TAR_NAME}"
    print(f"[sftp] 上传 {tar_path} -> {remote_tar}")

    def _progress(done, total):
        pct = (done / total * 100) if total else 0
        print(f"\r  进度: {pct:.1f}%  ({done/1024/1024:.1f}/{total/1024/1024:.1f} MB)", end="")

    sftp.put(str(tar_path), remote_tar, callback=_progress)
    print()
    sftp.close()

    print("[ssh] 在服务器上解包...")
    ssh.exec_command(f"mkdir -p {remote_dir}/..")
    ssh.exec_command(f"cd {remote_dir}/.. && rm -rf search_service_deploy && tar -xzf {remote_dir}/{TAR_NAME}")
    # 简单实现：实际推荐用 sftp.put 直接同步整个目录，此处先简化
    ssh.close()
    print(f"[OK] 已上传到 {user}@{host}:{remote_dir}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-only", action="store_true", help="仅打包不上传")
    ap.add_argument("--host", default="81.70.191.196")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default=os.environ.get("SSHPASS", ""))
    ap.add_argument("--remote-dir", default="/root/mingqiang")
    args = ap.parse_args()

    tar_path = Path(__file__).resolve().parent / TAR_NAME
    make_tar_gz(ROOT, tar_path)

    if args.local_only:
        print(f"[local-only] 跳过 SSH 上传，包：{tar_path}")
        return

    if not args.password:
        print("[WARN] 未提供密码：请用环境变量 SSHPASS=<pwd> 或 --password <pwd>")
        print("WARN: 退化为仅打包模式")
        return

    upload_ssh(tar_path, args.host, args.port, args.user, args.password, args.remote_dir)


if __name__ == "__main__":
    main()
