#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;
//g++ `pkg-config --cflags opencv4` `pkg-config --libs opencv4` rs.cpp -o rs

//int main( int argc, const char** argv )
int main()
{
   VideoCapture capture(CAP_INTELPERC);
   for(;;)
   {
      //double min;
      //double max;

      Mat depthMap;
      Mat image;
      Mat irImage;
      Mat adjMap;

      capture.grab();
   
      capture.retrieve( depthMap, CAP_INTELPERC_DEPTH_MAP );
      capture.retrieve(    image, CAP_INTELPERC_IMAGE );
      capture.retrieve(  irImage, CAP_INTELPERC_IR_MAP);

      //cv::minMaxIdx(depthMap, &min, &max);
      //depthMap.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
      normalize(depthMap, adjMap, 0, 255, NORM_MINMAX, CV_8UC1); 
      applyColorMap(adjMap, adjMap, COLORMAP_JET);

      imshow("RGB", image);
      imshow("IR", irImage);
      imshow("DEPTH", adjMap);
      if( waitKey( 30 ) >= 0 )
         break;
   }
   return 0;
}

