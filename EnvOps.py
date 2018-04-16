import urllib
import time
import os,sys
import numpy as np
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys

class game2048:
    cards = {"tile-" + str(2 ** i): 2 ** i for i in range(1, 12)}
    coordinates = {"tile-position-" + str(i) + '-' + str(j): (j - 1, i - 1) for i in range(1, 5) for j in range(1, 5)}
    UP = Keys.ARROW_UP
    LEFT = Keys.ARROW_LEFT
    DOWN = Keys.ARROW_DOWN
    RIGHT = Keys.ARROW_RIGHT
    keys = [UP, LEFT, DOWN, RIGHT]

class env:
    def __init__(self,driver):
        self.driver = driver
        self.site = 'https://gabrielecirulli.github.io/2048/'

    def activate(self):
        self.driver.get(self.site)

    def state(self):
        s = np.ones(shape=(4, 4), dtype=np.int)
        while True:
            try:
                information = [format(i.get_attribute("class")).split(" ")[1:3] for i in
                               self.driver.find_elements_by_class_name("tile")]
                for grid in information:
                    (c, r) = game2048.coordinates[grid[1]]
                    # print((c,r))
                    s[c, r] = game2048.cards[grid[0]]
                break
            except:
                pass
        return s

    def action(self,Key):
        Grid = self.driver.find_element_by_tag_name('body')
        Grid.send_keys(Key)

    def reward(self):
        score =self.driver.find_element_by_class_name('score-container')
        try:
            addition = score.find_element_by_class_name('score-addition')
            addition = np.int(addition.text[1:])
        except:
            addition = 0
        return addition

    def retry(self):
        try:
            end_bottom = self.driver.find_element_by_class_name('retry-button')
            end_bottom.click()
        except:
            pass



