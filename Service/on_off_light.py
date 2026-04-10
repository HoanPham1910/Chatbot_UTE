
import time
import hmac
import hashlib
import requests
import json

# ===== CONFIG =====
CLIENT_ID = "avscr3e833g3svp9e5xe"
SECRET = "ac010ce2046747d88b9f8ad7e8df81ea"
DEVICE_ID = "ebd5fe3bb68aa5d87a7yus"

BASE_URL = "https://openapi.tuyaus.com"


class TuyaController:

    def __init__(self):
        self.access_token = None
        self.token_expire = 0

    # =========================
    # SIGN TOKEN REQUEST
    # =========================
    def _sign_token(self, t):
        sign = hmac.new(
            SECRET.encode(),
            (CLIENT_ID + t).encode(),
            hashlib.sha256
        ).hexdigest().upper()
        return sign

    # =========================
    # GET NEW TOKEN
    # =========================
    def get_token(self):

        path = "/v1.0/token?grant_type=1"
        method = "GET"
        body = ""

        t = str(int(time.time() * 1000))

        body_hash = hashlib.sha256(body.encode()).hexdigest()

        string_to_sign = (
            method + "\n" +
            body_hash + "\n\n" +
            path
        )

        sign_str = CLIENT_ID + t + string_to_sign

        sign = hmac.new(
            SECRET.encode(),
            sign_str.encode(),
            hashlib.sha256
        ).hexdigest().upper()

        headers = {
            "client_id": CLIENT_ID,
            "sign": sign,
            "t": t,
            "sign_method": "HMAC-SHA256"
        }

        url = BASE_URL + path

        r = requests.get(url, headers=headers).json()

        print("TOKEN RESPONSE:", r)

        if not r["success"]:
            raise Exception(r)

        self.access_token = r["result"]["access_token"]
        self.token_expire = time.time() + r["result"]["expire_time"] - 60

        print("✅ Token OK")

    # =========================
    # CHECK TOKEN
    # =========================
    def ensure_token(self):
        if time.time() > self.token_expire:
            self.get_token()

    # =========================
    # SIGN API REQUEST
    # =========================
    def _sign_request(self, method, path, body):

        self.ensure_token()

        t = str(int(time.time() * 1000))

        body_hash = hashlib.sha256(body.encode()).hexdigest()

        string_to_sign = method + "\n" + body_hash + "\n\n" + path

        sign_str = CLIENT_ID + self.access_token + t + string_to_sign

        sign = hmac.new(
            SECRET.encode(),
            sign_str.encode(),
            hashlib.sha256
        ).hexdigest().upper()

        return sign, t

    # =========================
    # SWITCH DEVICE
    # =========================
    def switch(self, on=True):

        path = f"/v1.0/devices/{DEVICE_ID}/commands"

        body = json.dumps({
            "commands": [
                {"code": "switch_1", "value": on}
            ]
        })

        sign, t = self._sign_request("POST", path, body)

        headers = {
            "client_id": CLIENT_ID,
            "access_token": self.access_token,
            "sign": sign,
            "t": t,
            "sign_method": "HMAC-SHA256",
            "Content-Type": "application/json"
        }

        r = requests.post(BASE_URL + path,
                          headers=headers,
                          data=body)

        print(r.json())