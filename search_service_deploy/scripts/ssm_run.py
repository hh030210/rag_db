# -*- coding: utf-8 -*-
"""\
ssm_run.py
==========
用 paramiko 远程执行一条命令，可重复调用：

    python ssm_run.py "cd /opt/search_service/deploy && docker compose ps"
    python ssm_run.py "cd /opt/search_service/deploy && bash deploy_server.sh"
"""
import os
import sys

import paramiko


def main():
    if len(sys.argv) < 4:
        print("用法: python ssm_run.py <host> <user> <pwd> <cmd...>")
        sys.exit(1)
    host = sys.argv[1]
    user = sys.argv[2]
    pwd = sys.argv[3]
    cmd = " ".join(sys.argv[4:])
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=pwd, timeout=30)
    print(f"[remote] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=600)
    print("--- STDOUT ---")
    for line in stdout:
        print(line, end="")
    print("--- STDERR ---")
    for line in stderr:
        print(line, end="")
    print(f"--- exit={stdout.channel.recv_exit_status()} ---")
    ssh.close()


if __name__ == "__main__":
    main()
