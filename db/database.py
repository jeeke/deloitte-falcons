from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result
from db.config import url,username,password

# IBM Cloudant Legacy authentication
client = Cloudant(username, password, url=url)

def connectDB():
    client.connect()
    # IAM Authentication (uncomment if needed, and comment out previous IBM Cloudant Legacy authentication section)
    # client = Cloudant.iam("<username>", "<apikey>")
    # client.connect()
    database_name = "falcons-db"
    my_database = client[database_name]
    if my_database.exists():
        print(f"'{database_name}' successfully connected.")

# sample_data = [
#     [1, "one", "boiling", 100],
#     [2, "two", "hot", 40],
#     [3, "three", "warm", 20],
#     [4, "four", "cold", 10],
#     [5, "five", "freezing", 0]
# ]

# # Create documents using the sample data.
# # Go through each row in the array
# for document in sample_data:
#     # Retrieve the fields in each row.
#     number = document[0]
#     name = document[1]
#     description = document[2]
#     temperature = document[3]

#     # Create a JSON document that represents
#     # all the data in the row.
#     json_document = {
#         "numberField": number,
#         "nameField": name,
#         "descriptionField": description,
#         "temperatureField": temperature
#     }

#     # Create a document using the Database API.
#     new_document = my_database.create_document(json_document)

#     # Check that the document exists in the database.
#     if new_document.exists():
#         print(f"Document '{number}' successfully created.")

# result_collection = Result(my_database.all_docs)

# print(f"Retrieved minimal document:\n{result_collection[0]}\n")

# result_collection = Result(my_database.all_docs, include_docs=True)
# print(f"Retrieved full document:\n{result_collection[0]}\n")

# # try:
# #     client.delete_database(database_name)
# # except CloudantException:
# #     print(f"There was a problem deleting '{database_name}'.\n")
# # else:
# #     print(f"'{database_name}' successfully deleted.\n")

# client.disconnect()
