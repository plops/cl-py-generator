import pandas as pd
import scipy.optimize
import numpy as np
import subprocess
import threading
import mss
import cv2
import time
from PIL import Image
from cv2 import imshow, destroyAllWindows, imread, waitKey, imwrite, setMouseCallback, circle, matchTemplate, minMaxLoc
from ppadb.client import Client

_code_git_version = "9402add364b4d7369be67297958ce8ccff9261a7"
_code_repository = "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"
_code_generation_time = "17:50:34 of Sunday, 2021-06-20 (GMT+1)"
start_time = time.time()
debug = True
scrcpy = subprocess.Popen([
    "scrcpy", "--always-on-top", "-m", "640", "-w", "--window-x", "0",
    "--window-y", "0"
])
if (debug):
    print("{} start scrcpy ".format(((time.time()) - (start_time))))
adb = Client(host="127.0.0.1", port=5037)
devices = adb.devices()
if (((0) == (len(devices)))):
    c
    print("no device attached")
    quit()
device = devices[0]
running = True


def touch_collect_suns():
    global running
    while (running):
        x1 = 150
        y1 = 100
        x2 = 630
        y2 = 200
        if (debug):
            print("{} swipe x1={} y1={} x2={} y2={}".format(
                ((time.time()) - (start_time)), x1, y1, x2, y2))


def tap_play():
    global running
    while (running):
        x1 = 740
        y1 = 870
        if (debug):
            print("{} tap x1={} y1={}".format(((time.time()) - (start_time)),
                                              x1, y1))


time.sleep(1)
if (debug):
    print("{} prepare screenshot capture ".format(
        ((time.time()) - (start_time))))
sct = mss.mss()
scr = sct.grab(dict(left=5, top=22, width=640, height=384))
img = np.array(scr)
imshow("output", img)
if (debug):
    print("{} open templates ".format(((time.time()) - (start_time))))
res = []
fn = "img/SALE.jpg"
SALE = imread(fn, cv2.IMREAD_COLOR)


def find_strength_SALE(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, SALE, cv2.TM_CCORR_NORMED)
    w, h, ch = SALE.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_SALE(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, SALE, cv2.TM_CCORR_NORMED)
    w, h, ch = SALE.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


SALE_x = 1904
SALE_y = 105
SALE_screen_x = 608
SALE_screen_y = 37


def tap_SALE():
    if (debug):
        print("{} tap SALE SALE_x={} SALE_y={}".format(
            ((time.time()) - (start_time)), SALE_x, SALE_y))


res.append(dict(x=1904, y=105, sx=608, sy=37, name=SALE, fn=fn))
fn = "img/back.jpg"
back = imread(fn, cv2.IMREAD_COLOR)


def find_strength_back(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, back, cv2.TM_CCORR_NORMED)
    w, h, ch = back.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_back(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, back, cv2.TM_CCORR_NORMED)
    w, h, ch = back.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


back_x = 80
back_y = 74
back_screen_x = (25.50)
back_screen_y = 25


def tap_back():
    if (debug):
        print("{} tap back back_x={} back_y={}".format(
            ((time.time()) - (start_time)), back_x, back_y))


res.append(dict(x=80, y=74, sx=(25.50), sy=25, name=back, fn=fn))
fn = "img/how_to_play.jpg"
how_to_play = imread(fn, cv2.IMREAD_COLOR)


def find_strength_how_to_play(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, how_to_play, cv2.TM_CCORR_NORMED)
    w, h, ch = how_to_play.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_how_to_play(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, how_to_play, cv2.TM_CCORR_NORMED)
    w, h, ch = how_to_play.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


how_to_play_x = 970
how_to_play_y = 893
how_to_play_screen_x = 312
how_to_play_screen_y = 288


def tap_how_to_play():
    if (debug):
        print("{} tap how_to_play how_to_play_x={} how_to_play_y={}".format(
            ((time.time()) - (start_time)), how_to_play_x, how_to_play_y))


res.append(dict(x=970, y=893, sx=312, sy=288, name=how_to_play, fn=fn))
fn = "img/hook1.jpg"
hook1 = imread(fn, cv2.IMREAD_COLOR)


def find_strength_hook1(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, hook1, cv2.TM_CCORR_NORMED)
    w, h, ch = hook1.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_hook1(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, hook1, cv2.TM_CCORR_NORMED)
    w, h, ch = hook1.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


hook1_x = 292
hook1_y = 1090
hook1_screen_x = (91.50)
hook1_screen_y = 351


def tap_hook1():
    if (debug):
        print("{} tap hook1 hook1_x={} hook1_y={}".format(
            ((time.time()) - (start_time)), hook1_x, hook1_y))


res.append(dict(x=292, y=1090, sx=(91.50), sy=351, name=hook1, fn=fn))
fn = "img/pause.jpg"
pause = imread(fn, cv2.IMREAD_COLOR)


def find_strength_pause(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, pause, cv2.TM_CCORR_NORMED)
    w, h, ch = pause.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_pause(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, pause, cv2.TM_CCORR_NORMED)
    w, h, ch = pause.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/Play.jpg"
Play = imread(fn, cv2.IMREAD_COLOR)


def find_strength_Play(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, Play, cv2.TM_CCORR_NORMED)
    w, h, ch = Play.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_Play(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, Play, cv2.TM_CCORR_NORMED)
    w, h, ch = Play.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/empty_tile_powerup.jpg"
empty_tile_powerup = imread(fn, cv2.IMREAD_COLOR)


def find_strength_empty_tile_powerup(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, empty_tile_powerup, cv2.TM_CCORR_NORMED)
    w, h, ch = empty_tile_powerup.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_empty_tile_powerup(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, empty_tile_powerup, cv2.TM_CCORR_NORMED)
    w, h, ch = empty_tile_powerup.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/lets_rock_2.jpg"
lets_rock_2 = imread(fn, cv2.IMREAD_COLOR)


def find_strength_lets_rock_2(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, lets_rock_2, cv2.TM_CCORR_NORMED)
    w, h, ch = lets_rock_2.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_lets_rock_2(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, lets_rock_2, cv2.TM_CCORR_NORMED)
    w, h, ch = lets_rock_2.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/lets_rock.jpg"
lets_rock = imread(fn, cv2.IMREAD_COLOR)


def find_strength_lets_rock(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, lets_rock, cv2.TM_CCORR_NORMED)
    w, h, ch = lets_rock.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_lets_rock(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, lets_rock, cv2.TM_CCORR_NORMED)
    w, h, ch = lets_rock.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/lets_rock_weapon_selection.jpg"
lets_rock_weapon_selection = imread(fn, cv2.IMREAD_COLOR)


def find_strength_lets_rock_weapon_selection(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, lets_rock_weapon_selection, cv2.TM_CCORR_NORMED)
    w, h, ch = lets_rock_weapon_selection.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_lets_rock_weapon_selection(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, lets_rock_weapon_selection, cv2.TM_CCORR_NORMED)
    w, h, ch = lets_rock_weapon_selection.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/lets_rock_arena_populated.jpg"
lets_rock_arena_populated = imread(fn, cv2.IMREAD_COLOR)


def find_strength_lets_rock_arena_populated(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, lets_rock_arena_populated, cv2.TM_CCORR_NORMED)
    w, h, ch = lets_rock_arena_populated.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_lets_rock_arena_populated(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, lets_rock_arena_populated, cv2.TM_CCORR_NORMED)
    w, h, ch = lets_rock_arena_populated.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/load_old_plants.jpg"
load_old_plants = imread(fn, cv2.IMREAD_COLOR)


def find_strength_load_old_plants(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, load_old_plants, cv2.TM_CCORR_NORMED)
    w, h, ch = load_old_plants.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_load_old_plants(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, load_old_plants, cv2.TM_CCORR_NORMED)
    w, h, ch = load_old_plants.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/placed_btm_shooter_on_powerup.jpg"
placed_btm_shooter_on_powerup = imread(fn, cv2.IMREAD_COLOR)


def find_strength_placed_btm_shooter_on_powerup(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, placed_btm_shooter_on_powerup,
                        cv2.TM_CCORR_NORMED)
    w, h, ch = placed_btm_shooter_on_powerup.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_placed_btm_shooter_on_powerup(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, placed_btm_shooter_on_powerup,
                        cv2.TM_CCORR_NORMED)
    w, h, ch = placed_btm_shooter_on_powerup.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/placed_tree.jpg"
placed_tree = imread(fn, cv2.IMREAD_COLOR)


def find_strength_placed_tree(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, placed_tree, cv2.TM_CCORR_NORMED)
    w, h, ch = placed_tree.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_placed_tree(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, placed_tree, cv2.TM_CCORR_NORMED)
    w, h, ch = placed_tree.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/plant_balls.jpg"
plant_balls = imread(fn, cv2.IMREAD_COLOR)


def find_strength_plant_balls(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, plant_balls, cv2.TM_CCORR_NORMED)
    w, h, ch = plant_balls.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_plant_balls(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, plant_balls, cv2.TM_CCORR_NORMED)
    w, h, ch = plant_balls.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/plant_btm_shooter.jpg"
plant_btm_shooter = imread(fn, cv2.IMREAD_COLOR)


def find_strength_plant_btm_shooter(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, plant_btm_shooter, cv2.TM_CCORR_NORMED)
    w, h, ch = plant_btm_shooter.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_plant_btm_shooter(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, plant_btm_shooter, cv2.TM_CCORR_NORMED)
    w, h, ch = plant_btm_shooter.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/plant_tree.jpg"
plant_tree = imread(fn, cv2.IMREAD_COLOR)


def find_strength_plant_tree(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, plant_tree, cv2.TM_CCORR_NORMED)
    w, h, ch = plant_tree.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_plant_tree(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, plant_tree, cv2.TM_CCORR_NORMED)
    w, h, ch = plant_tree.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/X.jpg"
X = imread(fn, cv2.IMREAD_COLOR)


def find_strength_X(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, X, cv2.TM_CCORR_NORMED)
    w, h, ch = X.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_X(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, X, cv2.TM_CCORR_NORMED)
    w, h, ch = X.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/leaf_to_collect.jpg"
leaf_to_collect = imread(fn, cv2.IMREAD_COLOR)


def find_strength_leaf_to_collect(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, leaf_to_collect, cv2.TM_CCORR_NORMED)
    w, h, ch = leaf_to_collect.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_leaf_to_collect(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, leaf_to_collect, cv2.TM_CCORR_NORMED)
    w, h, ch = leaf_to_collect.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/sun_to_collect.jpg"
sun_to_collect = imread(fn, cv2.IMREAD_COLOR)


def find_strength_sun_to_collect(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, sun_to_collect, cv2.TM_CCORR_NORMED)
    w, h, ch = sun_to_collect.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_sun_to_collect(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, sun_to_collect, cv2.TM_CCORR_NORMED)
    w, h, ch = sun_to_collect.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/leaf_green_empty.jpg"
leaf_green_empty = imread(fn, cv2.IMREAD_COLOR)


def find_strength_leaf_green_empty(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, leaf_green_empty, cv2.TM_CCORR_NORMED)
    w, h, ch = leaf_green_empty.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_leaf_green_empty(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, leaf_green_empty, cv2.TM_CCORR_NORMED)
    w, h, ch = leaf_green_empty.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/leaf_green_notempty.jpg"
leaf_green_notempty = imread(fn, cv2.IMREAD_COLOR)


def find_strength_leaf_green_notempty(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, leaf_green_notempty, cv2.TM_CCORR_NORMED)
    w, h, ch = leaf_green_notempty.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_leaf_green_notempty(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, leaf_green_notempty, cv2.TM_CCORR_NORMED)
    w, h, ch = leaf_green_notempty.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/continue_current_reward_streak.jpg"
continue_current_reward_streak = imread(fn, cv2.IMREAD_COLOR)


def find_strength_continue_current_reward_streak(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, continue_current_reward_streak,
                        cv2.TM_CCORR_NORMED)
    w, h, ch = continue_current_reward_streak.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_continue_current_reward_streak(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, continue_current_reward_streak,
                        cv2.TM_CCORR_NORMED)
    w, h, ch = continue_current_reward_streak.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
fn = "img/continue_got_gauntlet.jpg"
continue_got_gauntlet = imread(fn, cv2.IMREAD_COLOR)


def find_strength_continue_got_gauntlet(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, continue_got_gauntlet, cv2.TM_CCORR_NORMED)
    w, h, ch = continue_got_gauntlet.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    return ma


def find_and_tap_continue_got_gauntlet(scr):
    imga = np.array(scr)
    img = imga[:, :, :-1]
    res = matchTemplate(img, continue_got_gauntlet, cv2.TM_CCORR_NORMED)
    w, h, ch = continue_got_gauntlet.shape
    mi, ma, miloc, maloc = minMaxLoc(res)
    center = ((np.array(maloc)) + ((((0.50)) * (np.array([h, w])))))
    circle(imga, tuple(center.astype(int)), 20, (
        0,
        0,
        255,
    ), 5)
    tap_android(center[0], center[1])
    return imga


# no x y coords
dft = pd.DataFrame(res)
if (debug):
    print("{} compute transform from screen to scrcpy window coordinates ".
          format(((time.time()) - (start_time))))


def trafo(pars):
    offset_x, offset_y, scale_x, scale_y = pars
    return np.concatenate([
        ((dft.sx) - (((scale_x) * (((dft.x) - (offset_x)))))),
        ((dft.sy) - (((scale_y) * (((dft.y) - (offset_y))))))
    ])


sol = scipy.optimize.least_squares(trafo, (
    (1.260),
    (-7.350),
    (0.320),
    (0.320),
),
                                   method="lm")
print(sol)
offset_x, offset_y, scale_x, scale_y = sol.x
dft["cx"] = ((scale_x) * (((dft.x) - (offset_x))))
dft["cy"] = ((scale_y) * (((dft.y) - (offset_y))))
if (debug):
    print(
        "{} compute transform from scrcpy to device coordinates. use this to generate tap positions "
        .format(((time.time()) - (start_time))))


def trafo_inv(pars):
    offset_x_inv, offset_y_inv, scale_x_inv, scale_y_inv = pars
    return np.concatenate([
        ((dft.x) - (((scale_x_inv) * (((dft.sx) - (offset_x_inv)))))),
        ((dft.y) - (((scale_y_inv) * (((dft.sy) - (offset_y_inv))))))
    ])


sol = scipy.optimize.least_squares(trafo_inv, (
    (-0.410),
    (2.350),
    (3.120),
    (3.120),
),
                                   method="lm")
print(sol)
offset_x_inv, offset_y_inv, scale_x_inv, scale_y_inv = sol.x
dft["cx_inv"] = ((scale_x_inv) * (((dft.sx) - (offset_x_inv))))
dft["cy_inv"] = ((scale_y_inv) * (((dft.sy) - (offset_y_inv))))


def tap_android(x1, y1):
    """transform screenshot coordinates and tap"""
    xx1 = ((scale_x_inv) * (((x1) - (offset_x_inv))))
    yy1 = ((scale_y_inv) * (((y1) - (offset_y_inv))))
    if (debug):
        print("{} tap x1={} y1={} xx1={} yy1={}".format(
            ((time.time()) - (start_time)), x1, y1, xx1, yy1))
    device.shell("input touchscreen tap {} {}".format(xx1, yy1))


def tap_direct(x1, y1):
    """tap directly in android coordinates"""
    if (debug):
        print("{} tap x1={} y1={}".format(((time.time()) - (start_time)), x1,
                                          y1))
    device.shell("input touchscreen tap {} {}".format(x1, y1))


fsm_state = 0
while (True):
    scr = sct.grab(dict(left=5, top=22, width=640, height=384))
    imga = np.array(scr)
    if (((fsm_state) == (0))):
        play_strength = find_strength_Play(scr)
        if (debug):
            print("{} menu play_strength={}".format(
                ((time.time()) - (start_time)), play_strength))
        if ((((0.980)) < (play_strength))):
            imga = find_and_tap_Play(scr)
            fsm_state = 1
    elif (((fsm_state) == (1))):
        st_old_plants = find_strength_load_old_plants(scr)
        if (debug):
            print("{} weapon-selection st_old_plants={}".format(
                ((time.time()) - (start_time)), st_old_plants))
        if ((((0.990)) < (st_old_plants))):
            imga = find_and_tap_load_old_plants(scr)
            fsm_state = 2
    elif (((fsm_state) == (2))):
        time.sleep(1)
        fsm_state = 3
    elif (((fsm_state) == (3))):
        s = find_strength_lets_rock_weapon_selection(scr)
        if (debug):
            print("{} weapon-approval s={}".format(
                ((time.time()) - (start_time)), s))
        if ((((0.930)) < (s))):
            imga = find_and_tap_lets_rock_weapon_selection(scr)
            fsm_state = 4
    elif (((fsm_state) == (4))):
        time.sleep(1)
        fsm_state = 5
    elif (((fsm_state) == (5))):
        s = find_strength_plant_btm_shooter(scr)
        if ((((0.990)) < (s))):
            imga = find_and_tap_plant_btm_shooter(scr)
            tap_direct(1100, 250)
        fsm_state = 6
    elif (((fsm_state) == (6))):
        s = find_strength_plant_btm_shooter(scr)
        if ((((0.990)) < (s))):
            imga = find_and_tap_plant_btm_shooter(scr)
            tap_direct(1250, 433)
        fsm_state = 7
    elif (((fsm_state) == (7))):
        s = find_strength_plant_btm_shooter(scr)
        if ((((0.990)) < (s))):
            imga = find_and_tap_plant_btm_shooter(scr)
            tap_direct(1250, 617)
        fsm_state = 8
    elif (((fsm_state) == (8))):
        s = find_strength_plant_btm_shooter(scr)
        if ((((0.990)) < (s))):
            imga = find_and_tap_plant_btm_shooter(scr)
            tap_direct(1250, 800)
        fsm_state = 9
    elif (((fsm_state) == (9))):
        s = find_strength_plant_btm_shooter(scr)
        if ((((0.990)) < (s))):
            imga = find_and_tap_plant_btm_shooter(scr)
            tap_direct(1100, 984)
        fsm_state = 10
    elif (((fsm_state) == (10))):
        s = find_strength_plant_tree(scr)
        if ((((0.990)) < (s))):
            imga = find_and_tap_plant_tree(scr)
            tap_direct(1400, 250)
        fsm_state = 11
    elif (((fsm_state) == (11))):
        s = find_strength_plant_tree(scr)
        if ((((0.990)) < (s))):
            imga = find_and_tap_plant_tree(scr)
            tap_direct(1400, 433)
        fsm_state = 12
    elif (((fsm_state) == (12))):
        s = find_strength_plant_tree(scr)
        if ((((0.990)) < (s))):
            imga = find_and_tap_plant_tree(scr)
            tap_direct(1400, 800)
        fsm_state = 13
    elif (((fsm_state) == (13))):
        s = find_strength_plant_tree(scr)
        if ((((0.990)) < (s))):
            imga = find_and_tap_plant_tree(scr)
            tap_direct(1400, 984)
        fsm_state = 14
    elif (((fsm_state) == (14))):
        time.sleep(1)
        fsm_state = 15
    elif (((fsm_state) == (15))):
        s = find_strength_lets_rock_arena_populated(scr)
        if (debug):
            print("{} start-battle s={}".format(((time.time()) - (start_time)),
                                                s))
        if ((((0.990)) < (s))):
            imga = find_and_tap_lets_rock_arena_populated(scr)
            fsm_state = 16
    elif (((fsm_state) == (16))):
        time.sleep(1)
        fsm_state = 16
    imshow("output", imga)
    cv2.moveWindow("output", 650, 400)
    if (((((cv2.waitKey(25)) & (255))) == (ord("q")))):
        destroyAllWindows()
        running = False
        break
scrcpy.terminate()
