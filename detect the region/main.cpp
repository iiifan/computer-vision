#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stack>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg, int dog, int cat);
bool Connected_Component(const cv::Mat& _binImg, cv::Mat& _lableImg, int num, vector<int>&numbers);

int main(int argc, char** argv)
{
    cv::Mat Img=cv::imread("/Users/tangyifan/aaa.bmp",0);    //read the img
    Mat new_Img;
    if(!Img.data)
    {
        cout<<"could not open"<<endl;
        return -1;
    }
    imshow("original Img",Img);
    Mat d_Img = Img.clone();

    
    int channels_1=0;    //plot the historgram
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
    imshow("2_1 Histogram_orignal", drawImg);
    
    
    int nRows=d_Img.rows;  //compute the num of the smallest pixel value
    int nCols=d_Img.cols;
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
    
    int threshold_value = 30;
    for(int i=0;i<nRows;i++)  //convert the img into binary img
    {
        p=d_Img.ptr<u_int8_t>(i);
        for(int j=0;j<nCols;++j)
        {
            if(p[j]>threshold_value)
                p[j]=255;
            else
                p[j]=0;
        }
    }
    imshow("2_1 binary image",d_Img);
    cout<<"threshold value is "<<threshold_value<<endl;
    for(int i=0;i<nRows;i++)
    {
        p=d_Img.ptr<u_int8_t>(i);
        for(int j=0;j<nCols;++j)
        {
            if(p[j]==255)
                p[j]=1;
            else
                p[j]=0;
        }
    }
    

    int num = 10;
    vector<int>numbers;
    Mat bin_Img = d_Img.clone();
    Mat labelImg;
    Connected_Component(bin_Img, labelImg, num, numbers);
    
    int pixel_num=numbers[0];  // compute the num of piexels of each area
    vector<int>count_lb(pixel_num,0);
    int *q;
    for(int i = 0; i<labelImg.rows; i++)
    {
        q=labelImg.ptr<int>(i);
        for(int j=0; j<labelImg.cols; j++)
        {
            for(int m=1; m<=pixel_num; m++)
            {
                if(q[j]==numbers[m])
                    count_lb[m]++;
            }
        }
    }
    
    int blue = 1;  //find the smallest and th largest area
    int red = 1;
    for(int i=2; i<=pixel_num; i++)
    {
        if(count_lb[i]>count_lb[red]) {red=i;}
        if(count_lb[i]<count_lb[blue]) {blue=i;}
    }
    cout<<blue<<" and "<< red<< endl;

    for(int x=1; x<=pixel_num; x++)
    {
        cout<<"the num of "<<numbers[x]<<" is "<<count_lb[x]<<endl;
    }

    
    cv::Mat color_Img ;
    icvprLabelColor(labelImg, color_Img, numbers[blue], numbers[red]) ;
    cv::imshow("colorImg", color_Img) ;
    cv::waitKey(0) ;
    
    return 0 ;
}

bool Connected_Component(const cv::Mat& bin_Img, cv::Mat& lable_Img, int num, vector<int>&numbers)
{
    if (bin_Img.empty() ||
        bin_Img.type() != CV_8UC1)
    {
        return false;
    }
    
    
    lable_Img.release() ;  //fist pass
    bin_Img.convertTo(lable_Img, CV_32SC1) ;
    
    int label = 1 ;
    std::vector<int> labelSet ;
    labelSet.push_back(0) ;   // background: 0
    labelSet.push_back(1) ;   // foreground: 1
    
    int rows = bin_Img.rows ;  //compare the up&left pixels value to give the data_curRow[j] value.
    int cols = bin_Img.cols;
    for (int i = 1; i < rows; i++)
    {
        int* data_preRow = lable_Img.ptr<int>(i-1) ;
        int* data_curRow = lable_Img.ptr<int>(i) ;
        for (int j = 1; j < cols; j++)
        {
            if (data_curRow[j] == 1)
            {
                std::vector<int> neighborLabels ;
                neighborLabels.reserve(2) ;
                int leftPixel = data_curRow[j-1] ;
                int upPixel = data_preRow[j] ;
                if ( leftPixel > 1)
                {
                    neighborLabels.push_back(leftPixel) ;
                }
                if (upPixel > 1)
                {
                    neighborLabels.push_back(upPixel) ;
                }
                
                if (neighborLabels.empty())
                {
                    labelSet.push_back(++label) ;  // assign to a new label
                    data_curRow[j] = label ;
                    labelSet[label] = label ;
                }
                else
                {
                    std::sort(neighborLabels.begin(), neighborLabels.end()) ; //connect the label in the same area
                    int smallestLabel = *min_element(neighborLabels.begin(), neighborLabels.end());
                    data_curRow[j] = smallestLabel ;
                    int max= *max_element(neighborLabels.begin(), neighborLabels.end());
                    if(smallestLabel<max)
                        labelSet[max] = smallestLabel;
                }
            }
        }
    }
    
    
    for (size_t i = 2; i < labelSet.size(); i++)  //to give the label in the same area with the same value
    {
        int curLabel = labelSet[i] ;
        int preLabel = labelSet[curLabel] ;
        while (preLabel != curLabel)
        {
            curLabel = preLabel ;
            preLabel = labelSet[preLabel] ;
        }
        labelSet[i] = curLabel ;
        
    }
    
    
    for (int i = 0; i < rows; i++)  //second pass, tranform the piexel values
    {
        int* data = lable_Img.ptr<int>(i) ;
        for (int j = 0; j < cols; j++)
        {
            int& pixelLabel = data[j] ;
            pixelLabel = labelSet[pixelLabel] ;
        }
    }
    

    std::sort(labelSet.begin(), labelSet.end());  //count the num of different areas
    numbers.push_back(0);
    numbers.push_back(0);
    int y=0;
    for(int x=0; x<labelSet.size(); x++)
    {
        if(labelSet[x]>numbers[y] && labelSet[x]>1)
        {
            numbers.push_back(labelSet[x]);
            y++;
            numbers[y]=labelSet[x];
        }
    }
    numbers[0]=y;
    for(int x=0; x<=y; x++)
    {
        cout<<x<<":"<<numbers[x]<<endl;
    }
    return true;
    
}



void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg, int dog, int cat)
{
    if (_labelImg.empty() ||
        _labelImg.type() != CV_32SC1)
    {
        return;
    }
    
    std::map<int, cv::Scalar> colors ;
    
    int rows = _labelImg.rows ;
    int cols = _labelImg.cols ;
    int min=dog;
    int max= cat;
    cout<<"oh my god"<<min<<" "<<max<<endl;
    _colorLabelImg.release() ;
    _colorLabelImg.create(rows, cols, CV_8UC3) ;
    _colorLabelImg = cv::Scalar::all(0) ;
    
    for (int i = 0; i < rows; i++)
    {
        const int* data_src = (int*)_labelImg.ptr<int>(i) ;
        uchar* data_dst = _colorLabelImg.ptr<uchar>(i) ;
        for (int j = 0; j < cols; j++)
        {
            int pixelValue = data_src[j] ;
            if (pixelValue == min)
            {
                    colors[pixelValue] = Scalar(255,0,0) ;
                cv::Scalar color = colors[pixelValue] ;
                *data_dst++   = color[0] ;
                *data_dst++ = color[1] ;
                *data_dst++ = color[2] ;
            }
            else if (pixelValue == max)
            {
                colors[pixelValue] = Scalar(0,0,255) ;
                cv::Scalar color = colors[pixelValue] ;
                *data_dst++   = color[0] ;
                *data_dst++ = color[1] ;
                *data_dst++ = color[2] ;
            }
            else if (pixelValue!=min && pixelValue!=max && pixelValue>0)
            {
                colors[pixelValue] = Scalar(0,255,0) ;
                cv::Scalar color = colors[pixelValue] ;
                *data_dst++   = color[0] ;
                *data_dst++ = color[1] ;
                *data_dst++ = color[2] ;
            }
            else
            {
                data_dst++ ;
                data_dst++ ;
                data_dst++ ;
            }
            
        }
    }
}

