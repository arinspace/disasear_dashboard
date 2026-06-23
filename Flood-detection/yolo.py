"""
YOLO Flood Detection Video Processor
====================================
Processes video files using YOLOv8 for flood detection and water level measurement.
Detects flood areas in video frames and calculates distance from water to a reference line.

Dependencies:
    - cv2: OpenCV for video processing
    - torch: PyTorch framework
    - ultralytics: YOLOv8 implementation
    - PIL: Image processing and rendering
    - shapely: Geometric operations for intersection detection
"""

import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Polygon
from array import array
import math
import numpy
import torch.serialization
from ultralytics.nn.tasks import SegmentationModel

def yolo(video_path, firstCoordinate_x, firstCoordinate_y, secondCoordinate_x, secondCoordinate_y, pixelsInAMeter, tipHeight, warningLevel):
    """
    Process video for flood detection using YOLOv8 and water level measurement.
    
    Args:
        video_path (str): Path to input video file
        firstCoordinate_x (int): X coordinate of reference line start point
        firstCoordinate_y (int): Y coordinate of reference line start point
        secondCoordinate_x (int): X coordinate of reference line end point
        secondCoordinate_y (int): Y coordinate of reference line end point
        pixelsInAMeter (float): Pixel-to-meter conversion ratio for distance calculation
        tipHeight (float): Reference height in meters for distance measurement
        warningLevel (float): Distance threshold (in meters) to trigger warning
    
    Returns:
        None: Outputs processed video to 'Output Video.mp4' and displays in window
    
    Processing Flow:
        1. Load YOLOv8 segmentation model
        2. Read video frames sequentially
        3. Detect flood/water areas using YOLO inference
        4. Calculate intersection with reference line
        5. Compute distance from water surface to reference point
        6. Draw warning indicators if distance exceeds threshold
        7. Write annotated frames to output video
    """
    myFont = ImageFont.truetype("arial.ttf", 45)
    distances = []

    # allowlist SegmentationModel
    torch.serialization.add_safe_globals([SegmentationModel])

    # Load the YOLOv8 model
    model = YOLO("best.pt")
    model.to('cpu') 
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the original video's frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the desired output frame rate
    output_fps = fps

    # Define the output video file name and VideoWriter object
    output_video_path = "Output Video.mp4"
    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (int(cap.get(3)), int(cap.get(4))))

    def find_intersection(annotation, line):
        intersection = annotation.intersection(line)
        if intersection != [(array('d'), array('d'))]:
            return [(intersection.xy)]
        else:
            return None

    def calculateDistance(x1, y1, x2, y2):
        distance = tipHeight - (math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))) / pixelsInAMeter
        d = float("{:.2f}".format(distance))
        return d

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Getting annotations from result
            segments = getattr(getattr(results[0], 'masks'), 'segments')[0]
            segmentsSize = int(segments.size / 2)
            segment = segments[0:segmentsSize]
            segmentString = "0"
            for i in range(segmentsSize):
                segmentCoordinates_x = format(segment[i][0], '.6f')
                segmentCoordinates_y = format(segment[i][1], '.6f')
                segmentString = segmentString + " " + str(segmentCoordinates_x) + " " + str(segmentCoordinates_y)

            segmentString = segmentString.strip().split(' ')
            polygon_vertices = []

            for i in range(1, len(segmentString), 2):
                x = int(float(segmentString[i]) * frame.shape[1])
                y = int(float(segmentString[i + 1]) * frame.shape[0])
                polygon_vertices.append((x, y))

            # Calculating intersections
            line_coords = [(firstCoordinate_x, firstCoordinate_y), (secondCoordinate_x, secondCoordinate_y)]

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            color_coverted = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_coverted)

            # Create a LineString object
            line = LineString(line_coords)
            if find_intersection(Polygon(polygon_vertices), line) != [(array('d'), array('d'))]:
                intersection_points = find_intersection(Polygon(polygon_vertices), line)
                intersection_x = intersection_points[0][0][0]
                intersection_y = intersection_points[0][1][0]
                distance = calculateDistance(intersection_x, intersection_y, firstCoordinate_x, firstCoordinate_y)
                distances.append(distance)

                updated_line_coords = [(firstCoordinate_x, firstCoordinate_y), (intersection_x, intersection_y)]
                updated_line = LineString(updated_line_coords)

                line_color = (0, 255, 0)
                line_width = 3

                draw = ImageDraw.Draw(pil_image)
                draw.line(updated_line.coords, fill=line_color, width=line_width)
                draw.text((1013, 134), str(distance), font=myFont, fill=(0, 0, 0))
                if distance >= warningLevel:
                    draw.text((936, 98), "WARNING!!!", font=myFont, fill=(255, 0, 0))

            else:
                draw = ImageDraw.Draw(pil_image)
                draw.text((936, 98), "SAFE", font=myFont, fill=(0, 255, 0))

            result = numpy.array(pil_image)
            color_coverted_result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            cv2.imshow("YOLOv8 Inference", color_coverted_result)
            output_video.write(color_coverted_result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Release the video capture and output video objects
    output_video.release()
    cap.release()
    cv2.destroyAllWindows()