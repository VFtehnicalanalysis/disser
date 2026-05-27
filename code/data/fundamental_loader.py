from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import os
download_dir = os.path.abspath('downloads')
if not os.path.exists(download_dir):
    os.makedirs(download_dir)
options = Options()
options.add_argument('--headless')
options.add_experimental_option('prefs', {'download.default_directory': download_dir, 'download.prompt_for_download': False, 'download.directory_upgrade': True, 'safebrowsing.enabled': True})
driver = webdriver.Chrome(options=options)
ticker = 'VKCO'
url = f'https://financemarker.ru/stocks/MOEX/{ticker}/reports/income/'
driver.get(url)
time.sleep(5)
download_button = driver.find_element(By.CSS_SELECTOR, 'i.mdi.mdi-download')
download_button.click()
time.sleep(10)
driver.quit()
