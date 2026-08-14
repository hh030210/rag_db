import os
import paramiko, time

HOST = os.getenv('DEPLOY_HOST', '127.0.0.1')
USER = os.getenv('DEPLOY_USER', 'root')
PASS = os.getenv('DEPLOY_PASSWORD', '')

if not PASS:
    raise SystemExit('请通过 DEPLOY_PASSWORD 环境变量提供 SSH 密码')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, 22, USER, PASS)

def run(cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    return code, out, err

# kill any existing server
run('pkill -f "uvicorn" 2>/dev/null; sleep 1')

print('=== starting server ===')
run('cd /root/app/core && nohup bash start.sh > /tmp/server.log 2>&1 &')
time.sleep(5)

# read startup log
code, out, err = run('cat /tmp/server.log')
print(out)

# check process
code, out, err = run('ps aux | grep uvicorn | grep -v grep')
print('[ps]', out)

# check port
code, out, err = run('ss -tlnp | grep uvicorn')
print('[port]', out)

ssh.close()
print('[done]')
