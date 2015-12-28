#include "header/cvproject.hpp"


int main(int argc, char *argv[])
{
    // arg -> 4
    // 1 == filename
    // 2 == destination
    // 3 == process type [ 0 == detect | 1 == process ]
    if(argc == 4) {
        if(argv[3][0] == '0'){ //detect face
            IplImage* rawImage = cvLoadImage(argv[1]);
            detectFace(rawImage);
            cvSaveImage(argv[2],rawImage);
        } else if(argv[3][0] == '1') {//process face
            IplImage* rawImage = cvLoadImage(argv[1]);
            IplImage* processedImage = getProcessedFace(rawImage);
            if(processedImage == NULL) return 0;
            cvSaveImage(argv[2],processedImage);
        } else if(argv[3][0] == '2'){//process face from pre-defined face region
            IplImage* rawImage = cvLoadImage(argv[1]);
            IplImage* processedImage = getProcessedFaceFromPredefinedFaceRegion(rawImage);
            if(processedImage == NULL) return 0;
            cvSaveImage(argv[2],processedImage);
        } else if(argv[3][0] == '3'){//process face from pre-defined face region
            IplImage* rawImage = cvLoadImage(argv[1]);
            IplImage* processedImage = edgeDetectSobelAllChannels(rawImage);
            if(processedImage == NULL) return 0;
            cvSaveImage(argv[2],processedImage);
        } else if(argv[3][0] == '4'){//process face from pre-defined face region
            IplImage* rawImage = cvLoadImage(argv[1]);
            IplImage* processedImage = IlluminationNormalize(rawImage);
            if(processedImage == NULL) return 0;
            cvSaveImage(argv[2],processedImage);
        }
    } else {
        printf( "\n"
                "------------------- Usage -------------------\n"
                "- Open program in the format\n"
                "- %s [SOURCE] [DESTINATION] [PROCESS_TYPE]\n"
                "- \n"
                "-     PROCESS_TYPE\n"
                "-       0 = Detect Face\n"
                "-       1 = Process Image\n"
                "-       2 = Process Image\n"
                "-       3 = Edge Detect\n"
                "-       4 = Illumination Normalize\n"
                "- \n"
                "---------------------------------------------\n", argv[0]);
    }
}
