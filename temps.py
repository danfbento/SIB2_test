import os
import mysql.connector
from mysql.connector import errorcode

# def get_ssl():file
#   ssl = os.path.join('.', 'ssl', 'DigiCertGlobalRootG2.crt.pem')
  
  
def return_results(results_dir):
    all_files = os.listdir(os.path.abspath(results_dir))
    data_files = list(filter(lambda file: file.endswith('.txt'), all_files))
    
    summary = []
    for txt in data_files:
        with open(os.path.join(results_dir,txt)) as f:
            for line in f:
                content = line.split()
                
                summary.append({
                    'image': txt,
                    'detected_class': content[0],
                    'detected_coord1': content[1],
                    'detected_coord2': content[2],
                    'detected_coord3': content[3],
                    'detected_coord4': content[4]
                })
    
    # Obtain connection string information from the portal
    config = {
      'host':'sib2-mysql-server.mysql.database.azure.com',
      'user':'icon_sib2@sib2-mysql-server',
      'password':'D02_YK29',
      'database':'SIB2DB',
      #'client_flags': [ClientFlag.SSL],
      'ssl_ca': str(os.path.join('.', 'deployassets', 'ssl', 'DigiCertGlobalRootG2.crt.pem'))
      #'ssl_verify_cert':'true'
    }
    
    # Construct connection string
    try:
      conn = mysql.connector.connect(**config)
      print("Connection established")
    except mysql.connector.Error as err:
      if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with the user name or password")
      elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
      else:
        print(err)
    else:
      cursor = conn.cursor()
      
      counter = 1
      total = len(summary)    
      for dictionary in summary:
        # Insert data into table
        cursor.execute("INSERT INTO model5_test_detections (image, detected_class,detected_coord1, detected_coord2, detected_coord3, detected_coord4) VALUES (%s, %s, %s, %s, %s, %s);",
                      (dictionary['image'], 
                       dictionary['detected_class'], 
                       dictionary['detected_coord1'], 
                       dictionary['detected_coord2'], 
                       dictionary['detected_coord3'], 
                       dictionary['detected_coord4']
                       )
                      )
        print(f"Detection {counter} from {total}." 
              f"Detected class = {dictionary['detected_class']}")
        counter = counter +1
      print("Inserted",total,"row(s) of data into the database")
      
      # Cleanup
      conn.commit()
      cursor.close()
      conn.close()
      print("Done.")

results_dir = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/inference/output'
return_results(results_dir)

###########################################################
#import mysql.connector
#from mysql.connector import errorcode
from datetime import datetime

dt = datetime.utcnow()

# Obtain connection string information from the portal
config = {
  'host':'sib2-mysql-server.mysql.database.azure.com',
  'user':'icon_sib2@sib2-mysql-server',
  'password':'D02_YK29',
  'database':'SIB2DB'
  #'client_flags': [ClientFlag.SSL],
  #'ssl_cert': '/var/wwww/html/DigiCertGlobalRootG2.crt.pem'
}

# Construct connection string
try:
   conn = mysql.connector.connect(**config)
   print("Connection established")
except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with the user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
else:
  cursor = conn.cursor()

  # Drop previous table of same name if one exists
#   cursor.execute("DROP TABLE IF EXISTS inventory;")
#   print("Finished dropping table (if existed).")

#   # Create table
#   cursor.execute(
#       "CREATE TABLE model5_test_detections (id serial PRIMARY KEY, image VARCHAR(50), detected_class INTEGER, detected_coord1 FLOAT, detected_coord2 FLOAT, detected_coord3 FLOAT, detected_coord4 FLOAT);"
#       )
#   print("Finished creating table.")

  # Insert some data into table
  cursor.execute("INSERT INTO model5_test_detections (image, detected_class,detected_coord1, detected_coord2, detected_coord3, detected_coord4) VALUES (%s, %s, %s, %s, %s, %s);",
                 ("test.txt", 1, 0.1, 0.1, 0.1, 0.1))
  print("Inserted",cursor.rowcount,"row(s) of data.")

  # Cleanup
  conn.commit()
  cursor.close()
  conn.close()
  print("Done.")

###########################################################
import base64
import json
import os
import cv2

img_path = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo/niuk_nie1_wv04_05042018_x832_y4992.png'
img_original = cv2.imread(img_path)
print(type(img_original))

# 8 20201030 = apparently img was sent and the load failed. Adapt datasets.py to decode
data = {}
with open(img_path, mode='rb') as imgfile:
    img = imgfile.read()
data['img'] = base64.encodebytes(img).decode("utf-8")

jsonimg = json.dumps(data)
#resp = requests.post(url=link, data=jsonimg, headers=headers)
#print(resp.text)

# receive back
load = json.loads(jsonimg)
imdata = base64.b64decode(load['img'])
img_np = cv2.imdecode(imdata, cv2.CV_LOAD_IMAGE_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

print(type(imdata))
img = cv2.imread(imdata)

os.path.isfile(imdata) # probably will never work

###########################################################
import base64
import json
import cv2
import requests

img_path = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo/niuk_nie1_wv04_05042018_x832_y4992.png'
link = 'http://51.11.39.149:80/api/v1/service/aks-mod5-test/score'
api_key = 'bO7sp1iNprvKohAvB1rVtDcdOpHo9BXh'
headers = {'Content-Type': 'application/json',
           'Authorization': ('Bearer ' + api_key)
           }

# encode
img = cv2.imread(img_path)
string = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
dict = {
    'img': string
}
jsonimg2 = json.dumps(dict, ensure_ascii=False, indent=4)
resp = requests.post(url=link, data=jsonimg2, headers=headers)
print(resp.text)
with open('C:/Users/Danilo.Bento/Desktop/temp/garbage/resp.txt', 'w') as outfile:
    outfile.write(resp.text)


with open('./0.json', 'w') as outfile:
    json.dump(dict, outfile, ensure_ascii=False, indent=4)

# dencode    
import base64
import json
import cv2
import numpy as np

response = json.loads(jsonimg2)
string = response['img']
jpg_original = base64.b64decode(string)
jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
img = cv2.imdecode(jpg_as_np, flags=1)
os.path.isfile(img) # probably will never work

cv2.imwrite('C:/Users/Danilo.Bento/Desktop/temp/garbage/0.png', img)

######################################################
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
files = [img_path]
images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
images




x = 'niuk_nie1_wv04_05042018_x832_y4992.png'
x = x.replace('.png', '.txt')