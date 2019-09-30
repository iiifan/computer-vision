#include<opencv2/opencv.hpp>
#include<iostream>
#include<fstream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<math.h>
#include<fstream>
#include<vector>

using namespace std;
using namespace cv;

void RGB_to_HSI(Mat &RGB, Mat &HSI, Mat &GREY);
void DCT(Mat &grey, Mat &freq);
void REMOVE(Mat &freq, Mat &d1, Mat &d2);
void IDCT1(Mat &redraw, Mat &freq);
void IDCT2(Mat &redraw, Mat &freq);


int main(){
    Mat RGB = imread("/Users/tangyifan/basel3.bmp");
    Mat HSI = Mat(RGB.size(),CV_8UC3);
    Mat GREY=Mat(HSI.size(),CV_8UC1);
    Mat FREQ=Mat(HSI.size(),CV_8UC1);
    Mat NEW1=Mat(HSI.size(),CV_8UC1);
    Mat NEW2=Mat(HSI.size(),CV_8UC1);
    RGB_to_HSI(RGB,HSI,GREY);
    DCT(GREY, FREQ);
    Mat D1=FREQ.clone();
    Mat D2=FREQ.clone();
    REMOVE(FREQ, D1, D2);
    IDCT1(NEW1,D1);
    IDCT2(NEW2,D2);
    
    
    waitKey(0);
    return 0;
}

void RGB_to_HSI(Mat &RGB, Mat &HSI, Mat &GREY)
{
    vector<Mat> channels;
    split(HSI, channels);
    Mat H_vec = channels.at(0);
    Mat S_vec = channels.at(1);
    Mat I_vec = channels.at(2);
    float H,S,I;
    
    for(int i=0; i<RGB.rows; i++)
    {
        for(int j=0; j<RGB.cols; j++)
        {
            int R=RGB.at<Vec3b>(i,j)(0);
            int G=RGB.at<Vec3b>(i,j)(1);
            int B=RGB.at<Vec3b>(i,j)(2);
            
            float num1=(0.5*((R-G)+(R-B)))/(sqrt((R-G)*(R-G)+(R-B)*(G-B)));
            float meow=sqrt((R-G)*(R-G)+(R-B)*(G-B));
            if(meow==0)
                H=0;
            else
            {
                float theta = acos(num1);
                if(B<=G)
                    H= theta;
                else
                    H= 360-theta;
            }
            H_vec.at<uchar>(i,j) = int(H*255/360);
            
            int min=R;
            if(min>G) min=G;
            if(min>B) min=B;
            if(R+G+B==0)
                S=0;
            else
                S=1-min/(R+G+B);
            S_vec.at<uchar>(i,j) = int(S*255);
            
            I=(R+G+B)/3;
            I_vec.at<uchar>(i,j) = int(I);
            GREY.at<uchar>(i,j) = int(I);
        }
    }
    
    merge(channels, HSI);
    imshow("f1", RGB);
    imshow("HSI_Img", HSI);
    imshow("I", GREY);
}

void DCT(Mat &grey, Mat &freq)
{
    freq.convertTo(freq, CV_32FC1);
    float pi=3.1415926;
    int N=8;
    float au,av;
    for(int u=0; u<496; u++)
    {
        for(int v=0; v<760; v++)
        {
            float sum=0;
            for(int x=0; x<N; x++)
            {
                int firstu=u/N*N;
                for(int y=0; y<N; y++)
                {
                    int firstv=v/N*N;
                    int pixel = grey.at<uchar>(x+firstu, y+firstv);
                    sum=sum+pixel*cos(((2*x+1)*(u%N)*pi)/(2*N))*cos(((2*y+1)*(v%N)*pi)/(2*N));
                }
            }
            if(u%N==0) au=sqrt(0.125); else au=sqrt(0.25);
            if(v%N==0) av=sqrt(0.125); else av=sqrt(0.25);
            freq.at<float>(u, v)=au*av*sum;
        }
    }
    imshow("F",freq);
}

void REMOVE(Mat &freq, Mat &d1, Mat &d2)
{
    d1.convertTo(freq, CV_32FC1);
    d2.convertTo(freq, CV_32FC1);
    for(int i=0; i<freq.rows; i++)
    {
        for(int j=0; j<freq.cols; j++)
        {
            if(i%8==0&&j%8==0)
            {continue;}
            else
            {d1.at<float>(i,j)=0;}
        }
    }
    imshow("D1",d1);
    for(int i=0; i<freq.rows; i++)
    {
        for(int j=0; j<freq.cols; j++)
        {
            if(i%8<3&&j%8<3)
                continue;
            else
                d2.at<float>(i,j)=0;
        }
    }
    imshow("D2",d2);
}

void IDCT1(Mat &redraw, Mat &freq) //for d1
{
    redraw.convertTo(redraw, CV_32FC1);
    float pi=3.1415926;
    int N=8;
    float au,av;
    for(int x=0; x<496; x++)
    {
        for(int y=0; y<760; y++)
        {
            float sum=0;
            for(int u=0; u<N; u++)
            {
                int firstx=x/N*N;
                for(int v=0; v<N; v++)
                {
                    if(u==0) au=sqrt(0.125); else au=sqrt(0.25);
                    if(v==0) av=sqrt(0.125); else av=sqrt(0.25);
                    int firsty=y/N*N;
                    int x1=x%N;
                    int y1=y%N;
                    sum=sum+au*av*freq.at<float>(u+firstx, v+firsty)*cos(((2*x1+1)*u*pi)/(2*N))*cos(((2*y1+1)*v*pi)/(2*N));
                    
                }
            }
            redraw.at<float>(x, y)=sum;
        }
    }
    redraw.convertTo(redraw, CV_8UC1);
    imshow("R1",redraw);
}

void IDCT2(Mat &redraw, Mat &freq)  //for d2
{
    redraw.convertTo(redraw, CV_32FC1);
    float pi=3.1415926;
    int N=8;
    float au,av;
    for(int x=0; x<496; x++)
    {
        for(int y=0; y<760; y++)
        {
            float sum=0;
            for(int u=0; u<N; u++)
            {
                int firstx=x/N*N;
                for(int v=0; v<N; v++)
                {
                    if(u==0) au=sqrt(0.125); else au=sqrt(0.25);
                    if(v==0) av=sqrt(0.125); else av=sqrt(0.25);
                    int firsty=y/N*N;
                    int x1=x%N;
                    int y1=y%N;
                    sum=sum+au*av*freq.at<float>(u+firstx, v+firsty)*cos(((2*x1+1)*u*pi)/(2*N))*cos(((2*y1+1)*v*pi)/(2*N));
                    
                }
            }
            redraw.at<float>(x, y)=sum;
        }
    }
    redraw.convertTo(redraw, CV_8UC1);
    imshow("R2",redraw);
}
