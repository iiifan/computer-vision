#include<opencv2/opencv.hpp>
#include<iostream>
#include<fstream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#define w 400


using namespace cv;
using namespace std;

int main()
{
    Mat Img=imread("/Users/tangyifan/House_width_times4.bmp",0); //read image and clone it
    if(!Img.data)
    {
        cout<<"could not open"<<endl;
        return -1;
    }
    imshow("1_3 original_Img",Img);
    
    Mat d_Img = Img.clone();
    
    int nRows=d_Img.rows;  //compute the num of the smallest piexl
    int nCols=d_Img.cols;
    int nSum= nRows*nCols;
    int min_num=0;
    u_int8_t *p;
    
    double minv= 0.0, maxv= 0.0;
    minMaxIdx(Img, &minv,&maxv);
    cout<< minv << " "<< maxv <<endl;
    for(int i=0;i<nRows;i++)
    {
        p=d_Img.ptr<u_int8_t>(i);
        for(int j=0;j<nCols;++j)
        {
            if(p[j]==minv)
                min_num=min_num+1;
        }
    }
    cout<< min_num<< endl;
    
    
    
    for(int i=0;i<nRows;i++)   //Transform a grey-scale image into its negative image
    {
        p=d_Img.ptr<u_int8_t>(i);
        for(int j=0;j<nCols;++j)
        {
            p[j]= 256-1-p[j];
        }
    }
    imshow("1_1 negative_Img",d_Img);
    
    
    char line_window[] = "1-1 transformation curve";  //plot the transfromation curve
    Mat lineImg = Mat::zeros( w, w, CV_8UC3 );
    int i;
    int x_1, x_2;
    for(i=1; i<255; i++)
    {
        int y_1 = lineImg.rows-(256-1-(i-1));
        x_1=i-1;
        int y_2 = lineImg.rows-(256-1-i);
        x_2=i;
        Point p11(x_1, y_1);
        Point p21(x_2, y_2);
        line(lineImg, p11, p21, Scalar(33,33,133),2);
        imshow(line_window, lineImg);
    }
    
    
    int channels_1=0;  //compute and plot the histogram
    MatND dstHist;
    int histSize[] = {256};
    float midRanges[] = {0, 256};
    const float *ranges[] = {midRanges};
    calcHist(&Img, 1, &channels_1, cv::Mat(), dstHist, 1, histSize, ranges);
    Mat drawImg = Mat::zeros(256,256, CV_8UC3);
    double HistMAxValue;
    minMaxLoc(dstHist, 0, &HistMAxValue, 0, 0);
    for(int i =0; i<256; i++)
    {
        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / HistMAxValue);
        line(drawImg, Point(i, drawImg.rows - 1), Point(i, drawImg.rows - 1 - value), Scalar(0, 0, 255));
    }
    imshow("2_2 Histogram_orignal", drawImg);
    
    MatND dstHist_2;
    calcHist(&d_Img, 1, &channels_1, cv::Mat(), dstHist_2, 1, histSize, ranges);
    double HistMAxValue_2;
    minMaxLoc(dstHist_2, 0, &HistMAxValue_2, 0, 0);
    Mat drawImg_2 = Mat::zeros(256,256, CV_8UC3);
    for(int i =0; i<256; i++)
    {
        int value = cvRound(dstHist_2.at<float>(i) * 256 * 0.9 / HistMAxValue_2);
        line(drawImg_2, Point(i, drawImg_2.rows - 1), Point(i, drawImg_2.rows - 1 - value), Scalar(0, 0, 255));
    }
    imshow("2_2 Histogram_nagetive", drawImg_2);

    
    Mat lut(1, 256, CV_8U);  //histogram equalization enhance
    float cdf[256] = {0};
    for(int i=0; i<256; i++)
    {
        if(i==0)
            cdf[i]=dstHist.at<float>(i);
        else
            cdf[i]=cdf[i-1]+dstHist.at<float>(i);
    }
    
    for(int i=0;i<nRows;i++)
    {
        p=Img.ptr<u_int8_t>(i);
        for(int j=0;j<nCols;++j)
        {
            int meow= p[j];
            int dog= cdf[meow];
            p[j]=cvRound((256-1)*(dog-min_num)/(nSum-min_num));
        }
    }
    imshow("2_3 enhanced image",Img);
    
    
    calcHist(&Img, 1, &channels_1, cv::Mat(), dstHist, 1, histSize, ranges);
    Mat drawImg_3 = Mat::zeros(256, 256, CV_8UC3);
    double HistMAxValue_3;
    minMaxLoc(dstHist, 0, &HistMAxValue_3, 0, 0);
    for(int i =0; i<256; i++)
    {
        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / HistMAxValue_3);
        line(drawImg_3, Point(i, drawImg_3.rows - 1), Point(i, drawImg_3.rows - 1 - value), Scalar(0, 0, 255));
    }
    imshow("2_3 histogram_enhanced", drawImg_3);
    
    waitKey(0);
    return 0;
}

