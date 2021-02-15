#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"

using namespace std;
using namespace cv;

cv::CascadeClassifier face_cascade;
cv::cuda::CascadeClassifier* face_cascade_cuda;

void load_classifier(string face_cascade_path, bool use_cuda=0){
        if(use_cuda){
                face_cascade_cuda = cuda::CascadeClassifier::create(face_cascade_path);
                cout << "using gpu model" << endl;
        }
        else face_cascade.load(face_cascade_path);
}

vector<Rect> detect_faces(const Mat &image, bool use_cuda=0){
        double scalingFactor = 1.2;// with 1.001,too much false positive
	int numberOfNeighbours = 3;

        Mat frame_gray;
        cvtColor(image, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        vector<Rect> faces;
        if(use_cuda){
                cout << "using gpu inference" << endl;
                face_cascade_cuda->setScaleFactor(scalingFactor);
                face_cascade_cuda->setMinNeighbors(numberOfNeighbours);
                // face_cascade_cuda->setMaxObjectSize();
                // face_cascade_cuda->setMinObjectSize();

                cuda::GpuMat image_gpu(frame_gray);
                cuda::GpuMat face_gpu_buf;

                face_cascade_cuda->detectMultiScale(image_gpu, face_gpu_buf);
                face_cascade_cuda->convert(face_gpu_buf, faces);
        }
        else{
                face_cascade.detectMultiScale(frame_gray, faces, 
                                              scalingFactor, numberOfNeighbours, 0);
        }

        return faces;
}

void display(const Mat &frame, const vector<Rect> &faces){
        cout << "there is " << faces.size() << " face detected" << endl;
        size_t i, faces_size = faces.size();
        for(i = 0; i < faces_size; i++){
                Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
                ellipse(frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 255, 255), 4);
                Mat faceROI = frame(faces[i]);
        }
        //-- Show what you got
        imshow( "Capture - Face detection", frame );
        waitKey(0);
}

const std::string get_keys(){
        const std::string keys =
    "{@input_image | data/test.png | input image filename }"
    "{@classifier_path | data/haarcascades/haarcascade_frontalface_alt2.xml | classifier path }"
    "{@use_cuda | 0 | if enable cuda }";

        return keys;
}

int main(int argc, char *argv[]) {
        CommandLineParser parser(argc, argv, get_keys());
        Mat img = imread(parser.get<String>("@input_image"));
        bool use_cuda = parser.get<bool>("@use_cuda");
        cout << "1" << endl;
        load_classifier(parser.get<String>("@classifier_path"), use_cuda);
        cout << "2" << endl;

        cv::resize(img, img, cv::Size(1024, 864));
        
        std::vector<Rect> faces = detect_faces(img, use_cuda);

        display(img, faces);
        imshow( "Capture - Face detection", img);
        waitKey(0);
}