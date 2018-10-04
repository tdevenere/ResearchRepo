import os
import numpy as np
from point import point
import pandas as pd
from math import inf, floor
import RNN_Feature_Extractor as fe


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
            points.append(point((float(sample[1]) + float(sample[3])) / 2, (float(sample[2]) + float(sample[4])) / 2,
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

def first_nonzero(points):
    for i in range(len(points)):
        if points[i].x != 0 or points[i].y != 0:
            return i
    return -1

# extractSpeeds
# Given the arrays of mousepoints and eyepoints along with the mouse and eye start time and direction
# Calculate the speed of both eyes and mouse. Also return the kind which indicates the region of the sample
# being in the starting region, ending region, or middle region using the kind which has the most points for this sample
def extractSpeeds(mousepoints, eyepoints, mousestart, eyestart, direction):
    mousedistance = 0
    mousetime = 0
    first = first_nonzero(mousepoints)
    lastpoint = mousepoints[first]
    lasttime = mousepoints[first].time
    mkinds = [0, 0, 0]
    for point in mousepoints[first+1:]:
        if non_zero(point):
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
    first = first_nonzero(eyepoints)
    lastpoint = eyepoints[first]
    lasttime = eyepoints[first].time
    ekinds = [0, 0, 0]
    for point in eyepoints[first+1:]:
        if non_zero(point):
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

    first = first_nonzero(mousepoints)
    lastpoint = mousepoints[first]
    for mousepoint in mousepoints[first+1:]:
        mouseCounts += 1
        mouseanglesum += mousepoint.angle(lastpoint, point(mousepoint.x, mousepoint.y + 1, mousepoint.time))

    first = first_nonzero(eyepoints)
    lastpoint = eyepoints[first]
    for eyepoint in eyepoints[first+1:]:
        eyeCounts += 1
        eyeanglesum += lastpoint.angle(eyepoint, point(lastpoint.x, lastpoint.y + 1, lastpoint.time))

    return div(mouseanglesum, mouseCounts), div(eyeanglesum, eyeCounts)


def extractInteriorAngles(mousepoints, eyepoints, direction):
    eyeanglesum = 0
    mouseanglesum = 0
    eyecounts = 0
    mousecounts = 0

    first = first_nonzero(mousepoints)
    second = first_nonzero(mousepoints[first:])
    lastlastpoint = mousepoints[first]
    lastpoint = mousepoints[second]
    for mousepoint in mousepoints[second+1:]:
        if mousepoint.x != 0 or mousepoint.y != 0:
            mouseanglesum += lastpoint.angle(mousepoint, lastlastpoint)
            mousecounts += 1

    first = first_nonzero(eyepoints)
    second = first_nonzero(eyepoints[first:])
    lastlastpoint = eyepoints[first]
    lastpoint = eyepoints[second]
    for eyepoint in eyepoints[second+1:]:
        if eyepoint.x != 0 or eyepoint.y != 0:
            eyeanglesum += lastpoint.angle(eyepoint, lastlastpoint)
            eyecounts += 1

    return div(mouseanglesum, mousecounts), div(eyeanglesum, eyecounts)


# OLD VERSION OF THIS. DO NOT USE
# def extractSeparation(mousepoints, eyepoints, direction):
#     eyeindex = 1
#     distance = 0
#     counts = 0
#     for mousepoint in mousepoints:
#         while eyepoints[eyeindex].time < mousepoint.time and eyeindex < len(eyepoints) - 1 and \
#                 (eyepoints[eyeindex].x != 0 or eyepoints[eyeindex].y != 0):
#             eyeindex += 1
#         if mousepoint.x != 0 or mousepoint.y != 0:
#             lpoint = eyepoints[eyeindex].lerp(eyepoints[eyeindex - 1], mousepoint.time, True)
#             distance += lpoint.euclidianDistance(mousepoint)
#             counts += 1
#
#     return div(distance, counts)


# Find the closest non-zero point to the provided index and return the index of that value.
# On ties, choose the smaller of the two if possible.
def find_closest(points, index):
    dist = 0
    while in_range(points, index, dist):
        if index - dist >= 0:
            p1 = points[index - dist]
        else:
            p1 = None
        if index + dist < len(points):
            p2 = points[index + dist]
        else:
            p2 = None

        if non_zero(p1):
            return index - dist
        if non_zero(p2):
            return index + dist
        dist += 1

    return -1


def in_range(a, index, dist):
    return index - dist < 0 and index + dist > len(a)


def non_zero(pt):
    nz = False
    if pt is not None:
        if pt.x != 0 or pt.y != 0:
            nz = True
    return nz

# Find the separation between the mouse and eye points.
# Since they do not line up exactly in this aligned version the approach is as follows:
# 1. Find the 'average' location of all eye points and mouse points separately.
# 2. Calculate the distance between the average points.
def extractSeparation(mousepoints, eyepoints, direction):

    eye_counts = 0
    mse_counts = 0
    eye_sum = point(0, 0)
    mse_sum = point(0, 0)

    for pt in eyepoints:
        if non_zero(pt):
            eye_sum = eye_sum.add(pt)
            eye_counts += 1

    for pt in mousepoints:
        if non_zero(pt):
            mse_sum = mse_sum.add(pt)
            mse_counts += 1

    eye_avg = point(div(eye_sum.x,  eye_counts), div(eye_sum.y, eye_counts))
    mse_avg = point(div(mse_sum.x, mse_counts), div(mse_sum.y, mse_counts))

    if non_zero(eye_avg) and non_zero(mse_avg):
        distance = eye_avg.euclidianDistance(mse_avg)
    else:
        distance = 0
    return distance

# Given all data points required to generate a single feature vector
# Find eye and mouse speeds, angles, separation between eye and mouse and the region in which the samples were found in.
def create_feature_vector(mousePoints, eyePoints, mouseStarttime, eyeStarttime, direction, samplenumber):

    # Calculate the Speed
    mouseSpeed, eyeSpeed, mouseregion, eyeregion = extractSpeeds(mousePoints, eyePoints, mouseStarttime, eyeStarttime,
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


def make_features(aligned, points_per_vector, direction, samplenumber):
    iters = floor(len(aligned)/points_per_vector)

    for i in range(iters):
        sample = aligned[(i * points_per_vector):((i+1) * points_per_vector)]
        eye_sample = [point.fromlist([x[3], x[4], x[0]]) for x in sample]
        mse_sample = [point.fromlist([x[1], x[2], x[0]]) for x in sample]

        eye_start = eye_sample[0].time
        mse_start = mse_sample[0].time
        yield create_feature_vector(mse_sample, eye_sample, mse_start, eye_start, direction, samplenumber)


def generate_pairs(eyepts, mousepts):
    start = min([eyepts[0].time, mousepts[0].time])
    end = max([eyepts[-1].time, mousepts[-1].time])
    num_eyes = len(eyepts)
    num_mse = len(mousepts)
    next_eye = 0
    next_mouse = 0

    while next_eye < num_eyes or next_mouse < num_mse:
        if next_mouse >= num_mse:
            mse = point(0, 0, inf)
        else:
            mse = mousepts[next_mouse]
        if next_eye >= num_eyes:
            eye = point(0, 0, inf)
        else:
            eye = eyepts[next_eye]
        time = min_time(eye, mse)

        if time == mse.time:
            msept = [mse.x, mse.y]
            next_mouse += 1
        else:
            msept = [0, 0]
        if time == eyepts[next_eye].time:
            eyept = [eye.x, eye.y]
            next_eye += 1
        else:
            eyept = [0, 0]
        frame = [time, msept[0], msept[1], eyept[0], eyept[1]]
        yield frame


def min_time(eyept, msept):
    return min(eyept.time, msept.time)


def align_times(eyefile, mousefile):
    eyestart, eyepts = extractEyePoints(eyefile)
    mousestart, mousepts = extractMousePoints(mousefile)

    aligned_pts = []
    for timestep in generate_pairs(eyepts, mousepts):
        aligned_pts.append(timestep)

    return aligned_pts


def main(num_points):

    # Destination Directory
    outputPath = os.path.join(os.getcwd(), "v4/32-Users/Users_time_aligned")

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    # Input Directory
    path = os.path.join(os.getcwd(), "v4/32-Users/Users")
    usernames = os.listdir(path)


    for username in usernames:
        df = pd.DataFrame()
        print(username)

        userpath = os.path.join(path, username)
        userfile = os.listdir(userpath)

        # For each eye file find match it up with the mouse file with the same
        # direction and sample number.
        for file in userfile:
            if 'eyeTrackData' in file:
                number = file.split("eyeTrackData_dir")[1][:-4]
                direction = int(number[0])
                samplenumber = int(number[1:])
                mousefile = "mouseData_dir" + str(direction) + str(samplenumber) + ".txt"
                if mousefile not in userfile:
                    print("\tFile name found " + mousefile)
                    continue
                userfile.remove(mousefile)

                aligned = align_times(os.path.join(userpath, file), os.path.join(userpath, mousefile))
                points_per_vector = fe.find_optimal_points(aligned, num_points)

                for feature_vec in make_features(aligned, points_per_vector, direction, samplenumber):
                    df = df.append(feature_vec)

        # Output into a CSV File
        filename = username + '-features.csv'
        full_path = os.path.join(outputPath, filename)
        df.to_csv(full_path, index=False)


if __name__ == "__main__":
    num_points = 20
    main(num_points)