import requests

from server.settings import settings


def ping_self():
    try:
        requests.get(settings.nuxt_public_api_base, timeout=5)
        print("Pinged self ✅")
    except Exception as e:
        print("Ping failed ❌", e)