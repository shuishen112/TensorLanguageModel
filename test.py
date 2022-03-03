from waybackpy import WaybackMachineSaveAPI

url = "https://github.com"
user_agent = "Mozilla/5.0 (Windows NT 5.1; rv:40.0) Gecko/20100101 Firefox/40.0"
save_api = WaybackMachineSaveAPI(url, user_agent)
save_api.save()
save_api.cached_save
save_api.timestamp()