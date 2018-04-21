import paramiko
from sshtunnel import SSHTunnelForwarder


REMOTE_SERVER_IP='117.48.195.186'
PRIVATE_SERVER_IP="172.16.16.32"

with SSHTunnelForwarder(
    (REMOTE_SERVER_IP, 443),
    ssh_username='dm',
    ssh_password='Vts^pztbvE339@Rw',

    remote_bind_address=(PRIVATE_SERVER_IP, 10000),
    local_bind_address=('127.0.0.1', 4300)
) as tunnel:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('127.0.0.1', 4300)
    # do some operations with client session
    client.close()

print('FINISH!')