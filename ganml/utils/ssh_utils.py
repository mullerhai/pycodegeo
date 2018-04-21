import  paramiko

from sshtunnel import SSHTunnelForwarder

class ssh_client:

  def __init__(self,host,user,pwd,port=2222):
    self.host=host
    self.user=user
    self.pwd=pwd
    self.port=port

  def tunnels(self):
    server=SSHTunnelForwarder(

      "172.16.16.32",
      ssh_username='',
      ssh_password='',
      remote_bind_address=('127.0.0.1',4300),
    )
    server.start()

  def conn(self):
    ssh=paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    try:
      ssh.connect(self.host,self.port,self.user,self.pwd,allow_agent=True)
      print("connect ssh  successfully")

      return ssh
    except:
      return None

    print("connect sucessfully")

  def exec(self,command,filepath=None,param=None):
    ssh=self.conn()
    print(command)
    stdin,stdout,stderr=ssh.exec_command(command)

    result=stdout.read()
    print(stderr.read())
    print(result)
    return result

if __name__ == '__main__':
    host='117.48.195.186'
    port=2222
    user='dm'
    pwd='Vts^pztbvE339@Rw'
    cli=ssh_client(host,user,pwd,port)
    res=cli.exec("/usr/bin/ls")
    print("ssh running")