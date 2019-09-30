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


int main(int argc, char** argv)
{
    cv::Mat Img=cv::imread("/Users/tangyifan/paper.bmp",0);    //read the img
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
    
    int threshold_value = 250;
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
                p[j]=255; //有改动
            else
                p[j]=0;
        }
    }
    
    
    //part 1_horizontal
    vector<int> meow_empty1(nRows,1) ;
    for(int i=0;i<170;i++)
    {
        p=d_Img.ptr<u_int8_t>(i);
        for(int j=0;j<nCols;++j)
        {
            if(p[j]==0)
            {
                meow_empty1.push_back(0);
                meow_empty1[i]=0;
            }
        }
    }
    cout<<"rows"<<nRows<<endl;
    cout<<"cols"<<nCols<<endl;

    
    //part 2_horizontal
    vector<int> meow_empty2(nRows,1) ;
    for(int i=170;i<nRows;i++)
    {
        p=d_Img.ptr<u_int8_t>(i);
        for(int j=0;j<442;++j)
        {
            if(p[j]==0)
            {
                meow_empty2.push_back(0);
                meow_empty2[i]=0;
            }
        }
    }
    
    //part 3_horizontal
    vector<int> meow_empty3(nRows,1) ;
    for(int i=170;i<nRows;i++)
    {
        p=d_Img.ptr<u_int8_t>(i);
        for(int j=442;j<nCols;++j)
        {
            if(p[j]==0)
            {
                meow_empty3.push_back(0);
                meow_empty3[i]=0;
            }
        }
    }
    
    //draw the horizontal line（from command line 138 to 266）
    // segment the part 3_horizontal
    vector<int> meow_rows(2,0) ;
    vector<int> meow_cols(2,0) ;
    u_int8_t *q;
    for(int i = 170; i<nRows; i++)
    {
        int flag=0;
        int meow=i;
        q=d_Img.ptr<u_int8_t>(i);
        for(int j=442; j<nCols; j++)
        {
            if(q[j]==0 && flag != 0)
            {
                meow_rows.push_back(j);
                meow_rows[1]=j;
                flag=2;
            }
            if(q[j]==0 && flag ==0)
            {
                meow_rows.push_back(j);
                meow_rows[0]=j;
                flag=1;
            }
            if(j==nCols-1&&flag==2&&meow_empty3[meow-1]&&meow_empty3[meow-2]&&meow_empty3[meow-3]&&meow_empty3[meow-4]&&meow_empty3[meow-5]&&meow_empty3[meow-6]&&meow_empty3[meow-7])
                for(int m= meow; m< meow+1; m++)
                {
                    p=d_Img.ptr<u_int8_t>(m);
                    for(int n=460;n<862;++n)
                    {
                        p[n]=0;
                    }
                    cout<<"the line is : "<< meow<< endl;
                }          if(j==nCols-1&&flag==2&&meow_empty3[meow+1]&&meow_empty3[meow+2]&&meow_empty3[meow+3]&&meow_empty3[meow+4]&&meow_empty3[meow+5]&&meow_empty3[meow+6]&&meow_empty3[meow+7])
                for(int m= meow; m< meow+1; m++)
                {
                    p=d_Img.ptr<u_int8_t>(m);
                    for(int n=460;n<862;++n)
                    {
                        p[n]=0;
                    }
                }
        }
    }
    
    
    // segment the part 2_horizontal
    for(int i = 170; i<nRows; i++)
    {
        int flag=0;
        int meow=i;
        q=d_Img.ptr<u_int8_t>(i);
        for(int j=0; j<443; j++)
        {
            if(q[j]==0 && flag != 0)
            {
                meow_rows.push_back(j);
                meow_rows[1]=j;
                flag=2;
            }
            if(q[j]==0 && flag ==0)
            {
                meow_rows.push_back(j);
                meow_rows[0]=j;
                flag=1;
            }
            if(j==442&&flag==2&&meow_empty2[meow-1]&&meow_empty2[meow-2]&&meow_empty2[meow-3]&&meow_empty2[meow-4]&&meow_empty2[meow-5]&&meow_empty2[meow-6]&&meow_empty2[meow-7])
                for(int m= meow; m< meow+1; m++)
                {
                    p=d_Img.ptr<u_int8_t>(m);
                    for(int n=26;n<440;++n)
                    {
                        p[n]=0;
                    }
                    cout<<"the cols is : "<< meow<< endl;

                }          if(j==442&&flag==2&&meow_empty2[meow+1]&&meow_empty2[meow+2]&&meow_empty2[meow+3]&&meow_empty2[meow+4]&&meow_empty2[meow+5]&&meow_empty2[meow+6]&&meow_empty2[meow+7])
                    for(int m= meow; m< meow+1; m++)
                    {
                        p=d_Img.ptr<u_int8_t>(m);
                        for(int n=26;n<440;++n)
                        {
                            p[n]=0;
                        }
                    }
        }
    }

    // segment the part 3_horizontal
    for(int i = 0; i<150; i++)
    {
        int flag=0;
        int meow=i;
        q=d_Img.ptr<u_int8_t>(i);
        for(int j=0; j<nCols; j++)
        {
            if(q[j]==0 && flag != 0)
            {
                meow_rows.push_back(j);
                meow_rows[1]=j;
                flag=2;
            }
            if(q[j]==0 && flag ==0)
            {
                meow_rows.push_back(j);
                meow_rows[0]=j;
                flag=1;
            }
            if(j==nCols-1&&flag==2&&meow_empty1[meow-1])
                for(int m= meow; m< meow+1; m++)
                {
                    p=d_Img.ptr<u_int8_t>(m);
                    for(int n=meow_rows[0];n<meow_rows[1];++n)
                    {
                        p[n]=0;
                    }
                    cout<<"the cols is : "<< meow<< endl;
                    
                }          if(j==nCols-1&&flag==2&&meow_empty1[meow+1])
                    for(int m= meow; m< meow+1; m++)
                    {
                        p=d_Img.ptr<u_int8_t>(m);
                        for(int n=meow_rows[0];n<meow_rows[1];++n)
                        {
                            p[n]=0;
                        }
                    }
        }
    }
    
    

    //draw the vertical line（from command line 270 to 266）
    // segment the part 3_horizontal
    Mat col_Img= d_Img.t();
    int Rows=col_Img.rows;
    int Cols=col_Img.cols;
    //part 1_vertical
    vector<int> dog_empty1(nRows,1) ;
    for(int i=0;i<Rows;i++)
    {
        p=col_Img.ptr<u_int8_t>(i);
        for(int j=0;j<170;++j)
        {
            if(p[j]==0)
            {
                dog_empty1.push_back(0);
                dog_empty1[i]=0;
            }
        }
    }
    cout<<"rows"<<Rows<<endl;
    cout<<"cols"<<Cols<<endl;
    
    
    //part 2_vertical
    vector<int> dog_empty2(Rows,1) ;
    for(int i=0;i<Rows;i++)
    {
        p=col_Img.ptr<u_int8_t>(i);
        for(int j=170;j<Cols;++j)
        {
            if(p[j]==0)
            {
                dog_empty2.push_back(0);
                dog_empty2[i]=0;
            }
        }
    }
    
    // segment the part 1_vertical
    for(int i = 0; i<Rows; i++)
    {
        int flag=0;
        int meow=i;
        q=col_Img.ptr<u_int8_t>(i);
        for(int j=0; j<170; j++)
        {
            if(q[j]==0 && flag != 0)
            {
                meow_rows.push_back(j);
                meow_rows[1]=j;
                flag=2;
            }
            if(q[j]==0 && flag ==0)
            {
                meow_rows.push_back(j);
                meow_rows[0]=j;
                flag=1;
            }
            if(j==170-1&&flag==2&&dog_empty1[meow-1])
                for(int m= meow; m< meow+1; m++)
                {
                    p=col_Img.ptr<u_int8_t>(m);
                    for(int n=meow_rows[0];n<meow_rows[1];++n)
                    {
                        p[n]=0;
                    }
                    cout<<"the cols is : "<< meow<< endl;
                    cout<<"the row is : "<<meow_rows[0] <<","<<meow_rows[1]<< endl;
                    
                }
            if(j==170-1&&flag==2&&dog_empty1[meow+1])
                    for(int m= meow; m< meow+1; m++)
                    {
                        p=col_Img.ptr<u_int8_t>(m);
                        for(int n=meow_rows[0];n<meow_rows[1];++n)
                        {
                            p[n]=0;
                        }
                    }
        }
    }
    
    
    // segment the part 2_vertical
    for(int i = 0; i<Rows; i++)
    {
        int flag=0;
        int meow=i;
        q=col_Img.ptr<u_int8_t>(i);
        for(int j=170; j<Cols; j++)
        {
            if(q[j]==0 && flag != 0)
            {
                meow_rows.push_back(j);
                meow_rows[1]=j;
                flag=2;
            }
            if(q[j]==0 && flag ==0)
            {
                meow_rows.push_back(j);
                meow_rows[0]=j;
                flag=1;
            }
            if(j==Cols-1&&flag==2&&(dog_empty2[meow-1]||dog_empty2[meow-2]||dog_empty2[meow-2]))
                for(int m= meow; m< meow+1; m++)
                {
                    p=col_Img.ptr<u_int8_t>(m);
                    for(int n=meow_rows[0];n<meow_rows[1];++n)
                    {
                        p[n]=0;
                    }
                    cout<<"the cols is : "<< meow<< endl;
                    cout<<"the row is : "<<meow_rows[0] <<","<<meow_rows[1]<< endl;
                    
                }
            if(j==Cols-1&&flag==2&&(dog_empty2[meow+1]||dog_empty2[meow+2]||dog_empty2[meow+2]))
                for(int m= meow; m< meow+1; m++)
                {
                    p=col_Img.ptr<u_int8_t>(m);
                    for(int n=meow_rows[0];n<meow_rows[1];++n)
                    {
                        p[n]=0;
                    }
                }
        }
    }
    
    Mat d= col_Img.t();

    cv::imshow("zhuanImg^2", d) ;
    cv::imwrite("/Users/tangyifan/result.bmp", d) ;


    cv::waitKey(0) ;
    
    return 0 ;
}




