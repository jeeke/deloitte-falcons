from cloudant.client import Cloudant

password = "698f0e6838ed46908129faf395629ac0bbf3f70df203dac4c316b288e5bddfb5"
url = "https://334dccb6-bf97-4845-9457-b17e39301d1a-bluemix:698f0e6838ed46908129faf395629ac0bbf3f70df203dac4c316b288e5bddfb5@334dccb6-bf97-4845-9457-b17e39301d1a-bluemix.cloudantnosqldb.appdomain.cloud"
username = "334dccb6-bf97-4845-9457-b17e39301d1a-bluemix"
# IBM Cloudant Legacy authentication
# Create client using auto_renew to automatically renew expired cookie auth
client = Cloudant(username,password,url=url,
                 connect=True,
                 auto_renew=True)