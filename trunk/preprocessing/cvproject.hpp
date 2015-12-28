/*
    Pre-defined header file
    Using in OpenCv Face Regcognition Project
    Author : Kasi Chonpimai
*/

#include<cv.h>
#include<highgui.h>
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<cstring>

#define MAXC(a,b) a>b?a:b
#define MINC(a,b) a<b?a:b

typedef struct pair_tag{
    CvPoint pt1, pt2;
    bool null;
} CVPair;

const char frontal_face_cascade_name[]="/usr/local/opencv/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml";
const char mouth_cascade_name[]="/usr/local/opencv/share/opencv/haarcascades/haarcascade_mcs_mouth.xml";
const char nose_cascade_name[]="/usr/local/opencv/share/opencv/haarcascades/haarcascade_mcs_nose.xml";
const char eye_cascade_name[]="/usr/local/opencv/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

inline uchar* element( IplImage*, int, int, int );
IplImage* edgeDetectSobelAllChannels( IplImage* );
IplImage* gradientImage( IplImage* );
IplImage* rotate( IplImage*, double, double, double, double );
IplImage* getRecSubImage( IplImage*, CvPoint, CvPoint );
IplImage* getProcessedFace( IplImage* );
CVPair detectFace( IplImage* );
CVPair detectEye( IplImage* );

CVPair detectFace( IplImage* img ){

    CVPair result;
    CvPoint* pt1 = &result.pt1;
    CvPoint* pt2 = &result.pt2;

    static CvMemStorage* storage = 0;
    static CvHaarClassifierCascade* cascade = 0;

    cascade = (CvHaarClassifierCascade*)cvLoad( frontal_face_cascade_name, 0, 0, 0 );
    storage = cvCreateMemStorage(0);
    cvClearMemStorage( storage );

    try {
        //Find faces in Picture
        CvSeq* faces = cvHaarDetectObjects( img, cascade, storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize((img->height)/8,(img->width)/8) );

           // Create a new rectangle for drawing the face
            CvRect* r = (CvRect*)cvGetSeqElem( faces, 0 );

            // Find the dimensions of the face
            pt1->x = r->x;
            pt2->x = r->x+r->width;
            pt1->y = r->y;
            pt2->y = r->y+r->height;

    } catch ( int ) { result.null = true; return result; }

    cvReleaseMemStorage(&storage);
    result.null = false;
    return result;
}

CVPair detectEye( IplImage* img ){

    CVPair result;
    CvPoint pt1, pt2, pt3, pt4;
    CvPoint* pt5 = &result.pt1;
    CvPoint* pt6 = &result.pt2;

    static CvMemStorage* storage = 0;
    static CvHaarClassifierCascade* cascade = 0;

    cascade = (CvHaarClassifierCascade*)cvLoad( eye_cascade_name, 0, 0, 0 );
    storage = cvCreateMemStorage(0);
    cvClearMemStorage( storage );

    try {
        //Find faces in Picture
        CvSeq* faces = cvHaarDetectObjects( img, cascade, storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, cvSize((img->height)/8,(img->width)/8) );

            // Create a new rectangle for drawing the face
            CvRect* r = (CvRect*)cvGetSeqElem( faces, 0 );

            // Find the dimensions of the face
            pt1.x = r->x;
            pt2.x = r->x+r->width;
            pt1.y = r->y;
            pt2.y = r->y+r->height;

            for(int i=1;i<(faces->total);i++){
                // Create a new rectangle for drawing the face
                r = (CvRect*)cvGetSeqElem( faces, i );

                // Find the dimensions of the face
                pt3.x = r->x;
                pt4.x = r->x+r->width;
                pt3.y = r->y;
                pt4.y = r->y+r->height;

                if( (pt1.x<=pt3.x && pt3.x<=pt2.x && pt1.y<=pt3.y && pt3.y<=pt2.y) ||
                    (pt1.x<=pt3.x && pt3.x<=pt2.x && pt1.y<=pt4.y && pt4.y<=pt2.y) ||
                    (pt1.x<=pt4.x && pt4.x<=pt2.x && pt1.y<=pt3.y && pt3.y<=pt2.y) ||
                    (pt1.x<=pt4.x && pt4.x<=pt2.x && pt1.y<=pt4.y && pt4.y<=pt2.y) ) continue;
                else break;
            }

            //Search for exact eye ball position

        if(pt1.x < pt3.x){


            pt5->y=(pt1.y+pt2.y)/2; pt5->x=(pt1.x+pt2.x)/2;

            pt6->y=(pt3.y+pt4.y)/2; pt6->x=(pt3.x+pt4.x)/2;

        }
        else{

            pt6->y=(pt1.y+pt2.y)/2; pt6->x=(pt1.x+pt2.x)/2;

            pt5->y=(pt3.y+pt4.y)/2; pt5->x=(pt3.x+pt4.x)/2;

        }

        int dx = (pt5->x)-(pt6->x);
        int dy = (pt5->y)-(pt6->y);

        if( ((int) sqrt( dx*dx+dy*dy )) < (img->width)/4 ){
                result.null = true;
                return result;
        }
    } catch ( int ) { result.null = true; return result; }

    cvReleaseMemStorage(&storage);
    result.null = false;
    return result;
}

IplImage* getRecSubImage( IplImage* img, CvPoint P1, CvPoint P2 ){

    /* sets the Region of Interest
    Note that the rectangle area has to be __INSIDE__ the image */
    cvSetImageROI(img, cvRect(P1.x, P1.y, P2.x-P1.x, P2.y-P1.y));

    /* create destination image
    Note that cvGetSize will return the width and the height of ROI */
    IplImage *dst = cvCreateImage(cvGetSize(img),
                               img->depth,
                               img->nChannels);

    /* copy subimage */
    cvCopy(img, dst, NULL);

    /* always reset the Region of Interest */
    cvResetImageROI(img);

    return dst;
}

IplImage* rotate( IplImage* src, double angle, double factor, double x, double y ){

    IplImage* dst = cvCreateImage( cvSize( (src->width), (src->height) ), src->depth, src->nChannels);

    double m[6];
    CvMat M = cvMat(2, 3, CV_64F, m);

    m[0] = (double) factor*cos(-angle*CV_PI/180.0);
    m[1] = (double) factor*sin(-angle*CV_PI/180.0);
    m[2] = x;
    m[3] = -m[1];
    m[4] = m[0];
    m[5] = y;

    cvGetQuadrangleSubPix( src, dst, &M);

    return dst;
}

IplImage* edgeDetectSobelAllChannels( IplImage* oriImg ){

    printf("1 | %d %d %d %d\n",(oriImg->width), (oriImg->height),(oriImg->depth), (oriImg->nChannels));

    IplImage* edgeImg = cvCreateImage( cvGetSize(oriImg), oriImg->depth, 1);
    //printf("1.2\n");

    //intitial Gx Gy Sobel Mask

    int GX[3][3], GY[3][3];

    GX[0][0] = -1; GX[0][1] = 0; GX[0][2] = 1;
    GX[1][0] = -2; GX[1][1] = 0; GX[1][2] = 2;
    GX[2][0] = -1; GX[2][1] = 0; GX[2][2] = 1;

    GY[0][0] =  1; GY[0][1] =  2; GY[0][2] =  1;
    GY[1][0] =  0; GY[1][1] =  0; GY[1][2] =  0;
    GY[2][0] = -1; GY[2][1] = -2; GY[2][2] = -1;
    //printf("1.3\n");

    printf("%d %d\n [%d]\n",edgeImg->width, edgeImg->height,cvGet2D(edgeImg,edgeImg->width-1, edgeImg->height-1).val[0]);

    for(int x=0;x<(100)-1;x++)for(int y=0;y<(103)-1;y++){cvSet2D(edgeImg,x,y,cvScalar(0.0));}
    //cvSaveImage("xxx.jpg",oriImg);
    //cvSaveImage("xxxz.jpg",edgeImg);
    //printf("2\n");
/*
    for(int c=0;c<(oriImg->nChannels);c++){
        for(int x=1;x<(oriImg->width)-1;x++)for(int y=1;y<(oriImg->height)-1;y++){

            int X=0, Y=0, T;

            // Approximate X Gradient
            for(int i=-1;i<=1;i++)for(int j=-1;j<=1;j++)
                X+=GX[i+1][j+1]*cvGet2D(oriImg,x+i,y+i).val[c];

            // Approximate Y Gradient
            for(int i=-1;i<=1;i++)for(int j=-1;j<=1;j++)
                Y+=GY[i+1][j+1]*cvGet2D(oriImg,x+i,y+i).val[c];

            // Approximate Total Gradient
            T=abs(X)+abs(Y);
            if(T>=255) { cvSet2D(edgeImg,x,y,cvRealScalar(255.0)); }
            else {double v = (double) MAXC(cvGet2D(edgeImg,x,y).val[0],T); cvSet2D(edgeImg,x,y,cvRealScalar(v));}
        }
    }*/

    return edgeImg;
}

IplImage* gradientImage( IplImage* oriImg ){

    IplImage* edgeImg = cvCreateImage( cvSize( (oriImg->width), (oriImg->height) ), oriImg->depth, 1);

    //intitial Gx Gy Sobel Mask

    int GX[3][3], GY[3][3];

    GX[0][0] = -1; GX[0][1] = 0; GX[0][2] = 1;
    GX[1][0] = -2; GX[1][1] = 0; GX[1][2] = 2;
    GX[2][0] = -1; GX[2][1] = 0; GX[2][2] = 1;

    GY[0][0] =  1; GY[0][1] =  2; GY[0][2] =  1;
    GY[1][0] =  0; GY[1][1] =  0; GY[1][2] =  0;
    GY[2][0] = -1; GY[2][1] = -2; GY[2][2] = -1;

    for(int x=0;x<(oriImg->width);x++)for(int y=0;y<(oriImg->height);y++) cvSet2D(edgeImg,x,y,cvScalar(0.0));

    for(int c=0;c<(oriImg->nChannels);c++){
        for(int x=1;x<(oriImg->width)-1;x++)for(int y=1;y<(oriImg->height)-1;y++){

            int X=0, Y=0, T;

            // Approximate X Gradient
            for(int i=-1;i<=1;i++)for(int j=-1;j<=1;j++)
                X+=GX[i+1][j+1]*cvGet2D(oriImg,x+i,y+i).val[c];

            // Approximate Y Gradient
            for(int i=-1;i<=1;i++)for(int j=-1;j<=1;j++)
                Y+=GY[i+1][j+1]*cvGet2D(oriImg,x+i,y+i).val[c];

            // Approximate Total Gradient
            T=(int) sqrt((double)X*X+Y*Y);
            //T=abs(X);
            if(T>=150) { cvSet2D(edgeImg,x,y,cvRealScalar(255.0)); }
            else {cvSet2D(edgeImg,x,y,cvScalar(0.0));}
        }
    }
    return edgeImg;
}

IplImage* getProcessedFace( IplImage* img ){

    const double EYEDIS = 50; // which is 1/2 of face width

    IplImage* grayImg = cvCreateImage(cvSize(img->width,img->height),img->depth,1);
	cvCvtColor( img, grayImg, CV_RGB2GRAY );

    CVPair faceCo = detectFace( grayImg );

    CvPoint* p1 = &faceCo.pt1;
    CvPoint* p2 = &faceCo.pt2;
    int lx=abs( (p2->x)-(p1->x) );
    int ly=abs( (p2->y)-(p1->y) );

    IplImage* faceInImage = cvCreateImage( cvSize( lx, ly ), img->depth, 3);

    //Copy face to faceInImage
        for(int i=0;i<ly;i++)for(int j=0;j<lx;j++)
            cvSet2D(faceInImage,i,j, cvGet2D(img,i+(p1->y),j+(p1->x)) );

    //Detect eyes
        CVPair eyeCo = detectEye( faceInImage );
        eyeCo.pt1.x+=faceCo.pt1.x;
        eyeCo.pt1.y+=faceCo.pt1.y;
        eyeCo.pt2.x+=faceCo.pt1.x;
        eyeCo.pt2.y+=faceCo.pt1.y;

    //
    IplImage* refinedImage = rotate( img,atan((double) (eyeCo.pt2.y-eyeCo.pt1.y)/(eyeCo.pt2.x-eyeCo.pt1.x))*180.0/CV_PI,
                                sqrt((double) (eyeCo.pt2.y-eyeCo.pt1.y)*(eyeCo.pt2.y-eyeCo.pt1.y)+(eyeCo.pt2.x-eyeCo.pt1.x)*(eyeCo.pt2.x-eyeCo.pt1.x))/EYEDIS,
                                (double) (eyeCo.pt2.x+eyeCo.pt1.x)/2,
                                (double) (eyeCo.pt2.y+eyeCo.pt1.y)/2    );

    //Get face from refinedImage
    CvPoint faceP1 = cvPoint((int)((img->width)/2-(EYEDIS)),(int)((img->height)/2-(EYEDIS*3/4)));
    CvPoint faceP2 = cvPoint((int)((img->width)/2+(EYEDIS)),(int)((img->height)/2+(EYEDIS*9/4)));
    IplImage* face = getRecSubImage( refinedImage, faceP1, faceP2 );


    IplImage* grayFaceImage = cvCreateImage( cvSize( (face->width), (face->height) ), face->depth, 1);
    cvCvtColor( face, grayFaceImage, CV_RGB2GRAY );
    //IplImage* hist = cvCreateImage( cvSize( (face->width), (face->height) ), face->depth, 1);
    //cvEqualizeHist( grayFaceImage, hist );

    /*/Show image
    cvShowImage( "face", face );
    cvShowImage( "gray", grayFaceImage );
    cvShowImage( "hist", hist );
    cvShowImage( "refinedImage", refinedImage );
    cvShowImage( "img", img );*/

    return grayFaceImage;
}

IplImage* getProcessedFaceFromPredefinedFaceRegion( IplImage* img ){

    //Optimization For Better Performance of getProcessedFace()

    const double EYEDIS = 50; // which is 1/2 of face width

    CVPair faceCo; //Dummy Declaration
    CvPoint* p1 = &faceCo.pt1;
    CvPoint* p2 = &faceCo.pt2;
    p1->x = 0;
    p1->y = 0;
    p2->x = (img->width)-1;
    p2->y = (img->height)-1;

    IplImage* faceInImage = img;

    //Detect eyes
        CVPair eyeCo = detectEye( faceInImage );
        if( (eyeCo.null) == true) return NULL;
        eyeCo.pt1.x+=(p1->x);
        eyeCo.pt1.y+=(p1->y);
        eyeCo.pt2.x+=(p1->x);
        eyeCo.pt2.y+=(p1->y);

    //
    IplImage* refinedImage = rotate( img,atan((double) (eyeCo.pt2.y-eyeCo.pt1.y)/(eyeCo.pt2.x-eyeCo.pt1.x))*180.0/CV_PI,
                                sqrt((double) (eyeCo.pt2.y-eyeCo.pt1.y)*(eyeCo.pt2.y-eyeCo.pt1.y)+(eyeCo.pt2.x-eyeCo.pt1.x)*(eyeCo.pt2.x-eyeCo.pt1.x))/EYEDIS,
                                (double) (eyeCo.pt2.x+eyeCo.pt1.x)/2,
                                (double) (eyeCo.pt2.y+eyeCo.pt1.y)/2    );

    //Get face from refinedImage
    CvPoint faceP1 = cvPoint((int)((img->width)/2-(EYEDIS)),(int)((img->height)/2-(EYEDIS)));
    CvPoint faceP2 = cvPoint((int)((img->width)/2+(EYEDIS)),(int)((img->height)/2+(EYEDIS*3/2)));
    IplImage* face = getRecSubImage( refinedImage, faceP1, faceP2 );

    IplImage* grayFaceImage = cvCreateImage( cvSize( (face->width), (face->height) ), face->depth, 1);
    cvCvtColor( face, grayFaceImage, CV_RGB2GRAY );
    //IplImage* hist = cvCreateImage( cvSize( (face->width), (face->height) ), face->depth, 1);
    //cvEqualizeHist( grayFaceImage, hist );

    return grayFaceImage;
}

IplImage* IlluminationNormalize(IplImage* img){

//Gamma Correction
    IplImage* gray = cvCreateImage( cvGetSize(img), img->depth, 1);
    cvCvtColor( img, gray, CV_RGB2GRAY );
    IplImage* trans = cvCreateImage(cvGetSize(img), 32, 1);
    IplImage* temp = cvCreateImage(cvGetSize(img), 32, 1);
    //Define variable
    int gammainverse = 2;
    //Gamma Correction
    cvConvertScale(gray, temp, 1.0/255, 0);
    cvPow(temp, temp, gammainverse);
    cvConvertScale(temp, trans, 255, 0);

//Difference of Gaussian
    IplImage* p1 = cvCreateImage( cvGetSize(img), 32, 1);
    IplImage* p2 = cvCreateImage( cvGetSize(img), 32, 1);
    cvSmooth( trans,p1,CV_GAUSSIAN, 7, 7 );
    cvSmooth( trans,p2,CV_GAUSSIAN, 9, 9 );
    cvSub( p1, p2, trans );
    IplImage* final = cvCreateImage(cvGetSize(img), 8, 1);
    cvConvertScale( trans,final,127.0,127.0 );

//Histrogram Equalize
    //cvSmooth( final,final,CV_GAUSSIAN, 3, 3 );
    cvEqualizeHist( final, final );
    cvThreshold( final, final, 30, 255, CV_THRESH_BINARY );

    return final;
}
