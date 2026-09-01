#!/usr/bin/env python3
"""Update nginx config to add /case_iter/ location."""
import re

NGINX_PATH = "/etc/nginx/conf.d/kskg.conf"
NEW_LOCATION = '''
        location /case_iter/ {
            proxy_pass http://127.0.0.1:8000;
            proxy_redirect off;
            proxy_set_header Host $http_host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_http_version 1.1;
            add_header Access-Control-Allow-Origin '*' always;
            add_header Access-Control-Allow-Methods 'GET POST OPTIONS' always;
            add_header Access-Control-Allow-Headers 'Content-Type Authorization' always;
            if ($request_method = 'OPTIONS') { return 204; }
        }
'''

with open(NGINX_PATH, "r", encoding="utf-8") as f:
    content = f.read()

if "/case_iter/" in content:
    print("Already has /case_iter/ location, skipping.")
else:
    # Insert after /healthz location block
    pattern = r"(location /healthz \{.*?\n\s+\})"
    m = re.search(pattern, content, re.DOTALL)
    if m:
        pos = m.end()
        content = content[:pos] + "\n" + NEW_LOCATION + "\n" + content[pos:]
        with open(NGINX_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Inserted /case_iter/ location after /healthz.")
    else:
        print("Could not find /healthz location block!")
        exit(1)

print("Testing nginx config...")
import subprocess
r = subprocess.run(["nginx", "-t"], capture_output=True, text=True)
print(r.stdout, r.stderr)
if r.returncode != 0:
    print("ERROR: nginx config invalid")
    exit(1)

print("Reloading nginx...")
r = subprocess.run(["nginx", "-s", "reload"], capture_output=True, text=True)
print(r.stdout, r.stderr)
print("Done.")
