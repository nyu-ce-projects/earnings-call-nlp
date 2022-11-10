import certifi
import json
import zipfile
from urllib.request import urlopen
from urllib.parse import urlencode

from config import Config


BASE_FMP_API_URL = "https://financialmodelingprep.com/api/v3"
BASE_GDRIVE_URL = "https://drive.google.com/uc"


def urlFetch(url, responseHeaders=False):
    print("Calling URL: {}".format(url))
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


def get_api_key():
    return Config()().get("Dataset").get("FMP_API_KEY")


def __checkKeys(keys_required, params):
    for key in keys_required:
            if params.get(key, None) == None:
                raise ValueError("params should have the keys: {}".format(keys_required))

def getFMPDataset(endpoint="earning_call_transcript", params={"symbol": "AAPL", "quarter": 3, "year": 2020}):
    params["apikey"] = get_api_key()
    if endpoint == "earning_call_transcript":
        keys_required = ["symbol", "quarter", "year"]
        __checkKeys(keys_required, params)
        url = BASE_FMP_API_URL + "/" + endpoint + "/" + str(params.pop("symbol"))
        url += "?" + urlencode(params)
        jsonData = urlFetch(url)
        return jsonData
    elif endpoint == "ratios-ttm":
        keys_required = ["symbol"]
        __checkKeys(keys_required, params)
        url = BASE_FMP_API_URL + "/" + endpoint + "/" + str(params.pop("symbol"))
        jsonData = urlFetch(url)
        return jsonData
    else:
        raise NotImplementedError("Endpoint '{}' not implemented.".format(endpoint))


def getGdriveDataset(fileid, path):
    params = {"id" : fileid, "export": "download"}
    url = BASE_GDRIVE_URL + "?" + urlencode(params)
    response = urlopen(url, cafile=certifi.where())
    final_url = response.geturl()
    new_response = urlopen(final_url, cafile=certifi.where())
    with open(path, "wb") as f:
        f.write(new_response.read())


def unzipFile(path, target_path):
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(target_path)
