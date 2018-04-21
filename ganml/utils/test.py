
from ganml.utils import   ftp_client
from ganml.utils.hive_utils import hive_client


def hellohive():
  host = '127.0.0.1'
  port = 4200
  username = 'zhuzheng'
  auth = 'LDAP'
  password = "abc123."
  database = 'fkdb'
  table = 'tab_client_label'
  paramDict = {'client_nmbr': 'AA75', 'batch': 'p1'}

  filedlist = ['gid', 'realname', 'card']
  mu =  hive_client(host, username, password, port)
  conn = mu.connhive(database, auth)
  cursor = conn.cursor()
  limit = 1000
  df2 = mu.query_selectFileds_Dataframe(conn, table, filedlist, paramDict, 1000)
  print(df2.shape)
  print(df2.head())
  print(df2.columns())

def helloftp():
  host = 'ftps.geotmt.com'
  port = 21
  user = 'zhuzheng'
  pwd = 'zzgeotmt.2'
  ip = '117.48.195.150'
  cli=ftp_client(host,user,pwd)
  fs= cli.login(2,True)
  return cli ,fs
 # path='haining'
  #cli.ftplistDir(fs,path)


  # server_path='haining/upload/'
  # downlaod_Severfile='AA62p1_yl_v2.3_20180410.rar'
  # new_localfil='AA62p1_yl_20180410.rar'
  # cli.ftpDownloadSeverFile(fs,server_path,downlaod_Severfile,new_localfil)
  #

  # upfile='/Users/geo/Downloads/AA62p1_yl_v2.3_20180410.rar'
  # sever_path='upload/'
  # new_severname='AA62p1_yl_20180410.rar'
  # cli.ftpUploadLocalFile(fs,upfile,sever_path,new_severname)
  #
  # fs.quit()


if __name__ == '__main__':
  cli,fs=helloftp()
  server_path = 'haining'
  upfilepath = '/Users/geo/Downloads/'
  upf1='AA18p7_GDscore_20180412.txt'
  upf2="AA78p1_GDscore_20180412.txt"
  upf3='AA77p1_GDscore_20180412.txt'
  cli.ftpUploadLocalFile(fs,upfilepath+upf1,server_path,upf1)
  cli.ftpUploadLocalFile(fs, upfilepath + upf2, server_path, upf2)
  cli.ftpUploadLocalFile(fs, upfilepath + upf3, server_path, upf3)
  fs.quit()

