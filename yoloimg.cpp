/*
 * wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

OpenCV >= 4.0.0 : g++ `pkg-config --cflags opencv4` `pkg-config --libs opencv4` yoloimg.cpp -o  yoloimg
OpenCV <  4.0.0 : g++ `pkg-config --cflags opencv` `pkg-config --libs opencv` yoloimg.cpp -o  yoloimg
*/

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"
#include <string>  

std::string keys =
    "{ help  h     | | Print help message. }"
    "{ image       |<none>| Path to input image file.  }"
    "{ device      |  0 | camera device number. }"
    "{ classes     | coco.names | Optional path to a text file with names of classes to label detected objects. }"
    "{ scale       | 0.00392 | Optional path to a text file with names of classes to label detected objects. }"
    "{ thr         | .5 | Confidence threshold. }"
    "{ width       | 416 | Confidence threshold. }"
    "{ height      | 416 | Confidence threshold. }"
    "{ model       | yolov3.weights | Model file yolo. }"
    "{ config      | yolov3.cfg | Config file of yolo. }"
    "{ nms         | .4 | Non-maximum suppression threshold. }";


using namespace cv;
using namespace dnn;
using namespace std;

float confThreshold, nmsThreshold;
std::vector<std::string> classes;

void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

std::vector<String> getOutputsNames(const Net& net);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run object detection deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String imageFile = parser.get<String>("image");
    confThreshold = parser.get<float>("thr");
    nmsThreshold = parser.get<float>("nms");
    float scale = parser.get<float>("scale");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    std::string modelPath = parser.get<String>("model");
    std::string configPath = parser.get<String>("config");

    // Open file with classes names.
    if (parser.has("classes"))
    {
        std::string file = parser.get<String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    // Load a model.
    Net net = readNet(modelPath, configPath);
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    // Create a window
    static const std::string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Process frames.
    Mat frame, blob;
    frame = imread(imageFile);
        if (frame.empty())
        {
            exit(0);
        }

        // Create a 4D blob from a frame.
        Size inpSize(inpWidth > 0 ? inpWidth : frame.cols,
                     inpHeight > 0 ? inpHeight : frame.rows);
        blobFromImage(frame, blob, scale, inpSize, Scalar(), false, false);

        net.setInput(blob);
        std::vector<Mat> outs;
        net.forward(outs, outNames);
        postprocess(frame, outs, net);

        imshow(kWinName, frame);
	waitKey(0);
    return 0;
}

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar( rand() % 255,rand() % 255,rand() % 255 ,255.0),3);
    //rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0),3);

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

