import os
import numpy as np
from point import point
import pandas as pd
import random
from math import floor

microToSecond = 1000000
trainAmount = 1
outOf = 2
buttons = [point(840, 525),
           point(840, 40),
           point(1183, 182),
           point(1325, 525),
           point(1183, 868),
           point(840, 1010),
           point(497, 868),
           point(355, 525),
           point(497, 182)]
# buttons = [point(960, 540),
#            point(960, 40),
#            point(1314, 186),
#            point(1460, 540),
#            point(1314, 894),
#            point(960, 1040),
#            point(606, 894),
#            point(460, 540),
#            point(606, 186)]
radius = 200


def div(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator / denominator


def extractEyePoints(path):
    points = []
    with open(path) as inputFile:
        lines = [value for value in inputFile.readlines() if value != '\n']
        starttime = int(lines[1].split(" ")[2])
        samples = lines[2:]
        for sample in samples:
            sample = sample.split()
            points.append(point((float(sample[1]) + float(sample[3])) / 2, (float(sample[3]) + float(sample[4])) / 2,
                                int(sample[0])))

    return starttime, points


def extractMousePoints(path):
    points = []
    with open(path) as inputFile:
        lines = [value for value in inputFile.readlines() if value != '\n']
        starttime = int(lines[1].split(" ")[2])
        samples = lines[2:]
        for sample in samples:
            sample = sample.split()
            points.append(point(float(sample[2]), float(sample[3]), int(sample[0])))

    return starttime, points


# extractSpeeds
# Given the arrays of mousepoints and eyepoints along with the mouse and eye start time and direction
# Calculate the speed of both eyes and mouse. Also return the kind which indicates the region of the sample
# being in the starting region, ending region, or middle region using the kind which has the most points for this sample
def extractSpeeds(mousepoints, eyepoints, mousestart, eyestart, direction):
    mousedistance = 0
    mousetime = 0
    lasttime = mousestart
    lastpoint = mousepoints[0]
    mkinds = [0, 0, 0]
    for point in mousepoints[1:]:
        if point.euclidianDistance(buttons[0]) < radius:
            mkinds[0] += 1
        elif point.euclidianDistance(buttons[direction]) < radius:
            mkinds[1] += 1
        else:
            mkinds[2] += 1
        mousedistance += point.euclidianDistance(lastpoint)
        mousetime += point.time - lasttime
        lastpoint = point
        lasttime = point.time

    eyeedistance = 0
    eyetime = 0
    lasttime = eyestart
    lastpoint = eyepoints[0]
    ekinds = [0, 0, 0]
    for point in eyepoints[1:]:
        if point.euclidianDistance(buttons[0]) < radius:
            ekinds[0] += 1
        elif point.euclidianDistance(buttons[direction]) < radius:
            ekinds[1] += 1
        else:
            ekinds[2] += 1
        eyeedistance += point.euclidianDistance(lastpoint)
        eyetime += point.time - lasttime
        lastpoint = point
        lasttime = point.time

    mkind = mkinds.index(max(mkinds))
    ekind = ekinds.index(max(ekinds))
    return div(microToSecond * mousedistance, mousetime), div(microToSecond * eyeedistance, eyetime), mkind, ekind


def extractVerticalOffsetAngles(mousepoints, eyepoints, direction):
    eyeanglesum = 0
    mouseanglesum = 0
    eyeCounts = 0
    mouseCounts = 0

    lastpoint = mousepoints[0]
    for mousepoint in mousepoints[1:]:
        mouseCounts += 1
        mouseanglesum += mousepoint.angle(lastpoint, point(mousepoint.x, mousepoint.y + 1, mousepoint.time))

    lastpoint = eyepoints[0]
    for eyepoint in eyepoints[1:]:
        eyeCounts += 1
        eyeanglesum += lastpoint.angle(eyepoint, point(lastpoint.x, lastpoint.y + 1, lastpoint.time))

    return div(mouseanglesum, mouseCounts), div(eyeanglesum, eyeCounts)


def extractInteriorAngles(mousepoints, eyepoints, direction):
    eyeanglesum = 0
    mouseanglesum = 0
    eyecounts = 0
    mousecounts = 0

    lastlastpoint = mousepoints[0]
    lastpoint = mousepoints[1]
    for mousepoint in mousepoints[2:]:
        mouseanglesum += lastpoint.angle(mousepoint, lastlastpoint)
        mousecounts += 1

    lastlastpoint = eyepoints[0]
    lastpoint = eyepoints[1]
    for eyepoint in eyepoints[2:]:
        eyeanglesum += lastpoint.angle(eyepoint, lastlastpoint)
        eyecounts += 1

    return div(mouseanglesum, mousecounts), div(eyeanglesum, eyecounts)


def extractSeparation(mousepoints, eyepoints, direction):
    eyeindex = 1
    distance = 0
    counts = 0
    for mousepoint in mousepoints:
        while eyepoints[eyeindex].time < mousepoint.time and eyeindex < len(eyepoints) - 1:
            eyeindex += 1
        lpoint = eyepoints[eyeindex].lerp(eyepoints[eyeindex - 1], mousepoint.time, True)
        distance += lpoint.euclidianDistance(mousepoint)
        counts += 1

    return div(distance, counts)


# Given all data points required to generate a single feature vector
# Find eye and mouse speeds, angles, separation between eye and mouse and the region in which the samples were found in.
def create_feature_vector(mousePoints, eyePoints, mouseStarttime, eyeStarttime, direction, samplenumber):

    # Calculate the Speed
    mouseSpeed, eyeSpeed, eyeregion, mouseregion = extractSpeeds(mousePoints, eyePoints, mouseStarttime, eyeStarttime,
                                         direction)
    # Calculate the angles
    mouseAngle, eyeAngle = extractVerticalOffsetAngles(mousePoints, eyePoints, direction)
    mouseIAngle, eyeIAngle = extractInteriorAngles(mousePoints, eyePoints, direction)

    # Calculate the separation
    separation = extractSeparation(mousePoints, eyePoints, direction)

    concat_results = [direction, samplenumber, separation, mouseSpeed, eyeSpeed, mouseAngle, eyeAngle,
                     mouseIAngle, eyeIAngle, mouseregion, eyeregion]
    col_titles = ['Direction', 'Sample', 'Sep', 'Mouse Speed', 'Eye Speed', 'Mouse Angle', 'Eye Angle', 'Mouse I-Angle',
                  'Eye I-Angle', 'Mouse Region', 'Eye Region']
    df = pd.DataFrame([concat_results], columns=col_titles)
    return df


# make_features
# Generator to extract a series of feature vectors from the given sample.
# There are more mouse points than eye points, so solve this:
#   1. Use the eye points to determine how many feature vectors can be created from the sample
#       using the number of points per vector that is passed as a parameter.
#   2. Since there are more mouse points, divide the total number of points, by the number of iters to determine
#       the number of mouse points per feature.
# Generator yields complete feature vector per iteration.
def make_features(mousePoints, eyePoints, points_per_vector, direction, samplenumber):
    iters = floor(len(eyePoints) / points_per_vector)
    num_mouse_pts = floor(len(mousePoints) / iters)

    for i in range(iters):
        eye_sample = eyePoints[(i * points_per_vector):((i+1) * points_per_vector)]
        mse_sample = mousePoints[(i * num_mouse_pts):((i+1) * num_mouse_pts)]

        eye_start = eye_sample[0].time
        mse_start = mse_sample[0].time
        yield create_feature_vector(mse_sample, eye_sample, mse_start, eye_start, direction, samplenumber)

    # Make one last feature with remaining datapoints
    try:
        eye_sample = eyePoints[(iters * points_per_vector):]
        mse_sample = mousePoints[(iters * num_mouse_pts):]
        eye_start = eye_sample[0].time
        mse_start = mse_sample[0].time
        yield create_feature_vector(mse_sample, eye_sample, mse_start, eye_start, direction, samplenumber)

    except IndexError:
        pass


def find_optimal_points(eyePoints, num_points):
    k_1 = num_points - 1
    k = num_points
    k_p1 = num_points + 1
    ks = [k_1, k, k_p1]
    lost_pts = [len(eyePoints) % k_1, len(eyePoints) % k, len(eyePoints) % k_p1]
    return ks[lost_pts.index(min(lost_pts))]


def main(num_points):

    # Destination Directory
    outputPath = os.path.join(os.getcwd(), "CNN-extractorOutput")

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    # Input Directory
    path = os.path.join(os.getcwd(), "users2Best")
    usernames = os.listdir(path)


    for username in usernames:
        df = pd.DataFrame()
        print(username)

        userpath = os.path.join(path, username)
        userfile = os.listdir(userpath)

        # For each eye file find match it up with the mouse file with the same
        # direction and sample number.
        for eyefile in userfile:
            number = eyefile.split("eyeTrackData_dir")[1][:-4]
            direction = int(number[0])
            samplenumber = int(number[1:])
            mousefile = "mouseData_dir" + str(direction) + str(samplenumber) + ".txt"
            if mousefile not in userfile:
                print("\tFile name found " + mousefile)
                continue
            userfile.remove(mousefile)

            # Get the eye and mouse information
            eyeStarttime, eyePoints = extractEyePoints(os.path.join(userpath, eyefile))
            mouseStarttime, mousePoints = extractMousePoints(os.path.join(userpath, mousefile))

            points_per_vector = find_optimal_points(eyePoints, num_points)
            for feature_vec in make_features(eyePoints, mousePoints, points_per_vector, direction, samplenumber):
                df = df.append(feature_vec)


        # Output into a CSV File
        filename = username + '-features.csv'
        full_path = os.path.join(outputPath, filename)
        df.to_csv(full_path, index=False)


if __name__ == "__main__":
    num_points = 10
    main(num_points)
