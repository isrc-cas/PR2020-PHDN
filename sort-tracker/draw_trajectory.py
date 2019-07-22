import os
import glob
import numpy as np
import cv2
import PIL.ImageColor as ImageColor

# colours = np.random.rand(32,3)'AliceBlue', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 
STANDARD_COLORS = [
    'Red', 'Green', 'SlateBlue', 'Cyan', 'LightBlue', 'Violet', 'Pink', 
    'Gold', 'Orange', 'Orchid',
    'Chartreuse', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Plum', 'PowderBlue', 'Purple',
     'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Wheat', 'White',
    'WhiteSmoke', 'YellowGreen'
]

# seqs = 'output/2019-1-19/*.txt'
jpg_path = '../data/hand-tracking/LISA_HT_TestSet/test_positive'
track_results = 'output/2019-1-19/'#'/new_home/mnt/64T/hand-tracking/LISA_HT_TestAnnotation/test_annotation'#
# ann_dir = 'trajetories_drawn_images/test_annotation'##
out_dir = 'trajetories_drawn_images/2019-1-19-pre'#'trajetories_drawn_images/2019-1-19-compare-with-gt'#'trajetories_drawn_images/test_annotation'#
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for seq in os.listdir(jpg_path):
    print('Processing %s' % seq)
    trajetories = {}
    image = cv2.imread(os.path.join(jpg_path, seq, '0000.png'))
    # image = cv2.imread(os.path.join(ann_dir, '%s.png'%seq))
    hands = np.loadtxt(os.path.join(track_results, '%s.txt' % seq), delimiter=',')
    for hand in hands:
        if hand[1] not in trajetories:
            trajetories[hand[1]] = [[hand[2]+hand[4]/2, hand[3]+hand[5]/2]]
        else:
            trajetories[hand[1]].append([hand[2]+hand[4]/2, hand[3]+hand[5]/2])
    start = 2
    for tr in sorted(list(trajetories.keys())):
        # color = colours2[int(tr%32),:] * 255
        color = STANDARD_COLORS[start]
        start += 1
        print(color)
        r, g, b = ImageColor.getrgb(color)
        pts = np.array(trajetories[tr], np.int32)
        cv2.polylines(image, [pts.reshape((-1, 1, 2))], False, (b,g,r), 2)
        for pt in pts:
            cv2.circle(image, tuple(pt), 3, (b,g,r))

    cv2.imwrite(os.path.join(out_dir, '%s.png' % (seq)), image)