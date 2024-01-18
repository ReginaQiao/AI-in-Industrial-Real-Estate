import datetime
import hashlib
import hmac
import json
from io import BytesIO
from time import sleep

import pandas as pd
import requests

ACC_KEY = "CDC60C1B-E462-4D81-9F21-EFFE755D3A0C"
ENC_KEY = "F5D863C8-7DDA-4BE5-802F-00F37F605EC6"
BASKET_NAME = "FLBUA"


def api_call(apiCommand, accKey, encKey, call_type="GET"):
    url = "https://api.economy.com/data/v1/" + apiCommand
    timeStamp = datetime.datetime.strftime(
        datetime.datetime.utcnow(), "%Y-%m-%dT%H:%M:%SZ"
    )
    payload = bytes(accKey + timeStamp, "utf-8")
    signature = hmac.new(bytes(encKey, "utf-8"), payload, digestmod=hashlib.sha256)
    head = {
        "AccessKeyId": accKey,
        "Signature": signature.hexdigest(),
        "TimeStamp": timeStamp,
    }
    sleep(1)
    if call_type == "POST":
        response = requests.post(url, headers=head)
    elif call_type == "DELETE":
        response = requests.delete(url, headers=head)
    else:
        response = requests.get(url, headers=head)

    return response


def test():
    # Identify the basket to execute and save ID
    baskets = pd.DataFrame(json.loads(api_call("baskets/", ACC_KEY, ENC_KEY).text))
    basketId = baskets.loc[baskets["name"] == BASKET_NAME, "basketId"].item()

    # Execute the particular basket using ID
    call = ("orders?type=baskets&action=run&id=" + basketId)
    order = api_call(call, ACC_KEY, ENC_KEY, call_type="POST")
    orderId = order.text[12:48]

    # Download output
    call = "orders/" + orderId
    processing_check = True
    while processing_check:
        sleep(5)
        status = api_call(call, ACC_KEY, ENC_KEY)
        processing_check = json.loads(status.content.decode('utf-8'))['processing']
    new_call = ("orders?type=baskets&id=" + basketId)
    get_basket = api_call(new_call, ACC_KEY, ENC_KEY)

    # Get excel document
    get_basket = pd.read_excel(BytesIO(get_basket.content))
    data_df = pd.DataFrame(get_basket)

    return data_df.to_excel(BASKET_NAME + '.xlsx')


if __name__ == "__main__":
    test()
