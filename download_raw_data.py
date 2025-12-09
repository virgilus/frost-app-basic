import os
import requests
import config as c

url = "https://www.data.gouv.fr/api/1/datasets/r/6989ed1a-8ffb-4ef9-b008-340327c99430"
response = requests.get(url)

if os.path.exists(c.RAW_DATA_PATH) is False:
    os.makedirs(c.RAW_DATA_PATH)

if response.status_code == 200:
    with open(os.path.join(c.RAW_DATA_PATH, c.CITY_FILENAME), 'wb') as file:
        file.write(response.content)
    print(f"Raw data downloaded and saved to {os.path.join(c.RAW_DATA_PATH, c.CITY_FILENAME)}")
else:
    print(f"Failed to download raw data. Status code: {response.status_code}")