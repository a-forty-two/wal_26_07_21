import pyautogui as gui
import time
from selenium import webdriver
gui.FAILSAFE = False

screenWidth, screenHeight = gui.size()

#stats

gui.moveTo(0, screenHeight)

gui.click()


gui.typewrite('chrome', interval=0.50)

gui.press('enter')

browser = webdriver.Chrome(executable_path="chromedriver.exe")
browser.get('https://www.facebook.com')



gui.press('enter')

gui.screenshot('001.png')
