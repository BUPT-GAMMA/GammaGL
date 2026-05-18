"""Execute commands on the remote server via SSH."""
import paramiko
import sys

HOST = '10.112.247.63'
PORT = 37096
USER = 'root'
PASSWORD = '123'
WORKDIR = '/ai/data/lyr/Roberta/GammaGL-main/'
CONDA_BIN = '/root/miniconda3/bin/conda'
CONDA_ENV = 'gammaglrgt'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

cmd = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read().strip()

# Source conda profile, activate env, cd to project
full_cmd = f'''source /root/miniconda3/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && cd {WORKDIR} && export TL_BACKEND=torch && {cmd}'''

stdin, stdout, stderr = client.exec_command(full_cmd)
out = stdout.read().decode()
err = stderr.read().decode()

if out:
    print(out)
if err:
    print(err, file=sys.stderr)

client.close()
