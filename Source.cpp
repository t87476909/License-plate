#include <opencv2\opencv.hpp>
#include <highgui.h>
#include <stdio.h>
#include <cstdio>
#include<fstream>
#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stack>
#include<sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/video/background_segm.hpp>
#pragma comment(lib,"libmysql.lib")
#include<Windows.h>
#include<mysql.h>

using namespace cv;
using namespace std;
using namespace ml;

Mat frame;                              // 取前景
Mat fgMaskMOG2;                         // 前景模板，通MOG2方法生成
Ptr<BackgroundSubtractor> pMOG2;        // MOG2 背景消除
char keyboard;                          // 按鍵暫停
const char* filename;

MYSQL my_connection;
MYSQL_RES *res;
MYSQL_ROW sqlrow;
void transform(const vector<Mat> &split, Mat &testData) //歸一化
{
	int test = 0;
	for (auto it = split.begin(); it != split.end(); it++)
	{
		Mat tmp;
		//cout << "test=" << test++ << endl; //圖片張數
		resize(*it, tmp, Size(144, 33));
		testData.push_back(tmp.reshape(0, 1));
	}
	testData.convertTo(testData, CV_32F);
}
void svm_train(Ptr<SVM> &mode2, Mat &trainData, Mat &trainLabels) //訓練SVM
{
	mode2->setType(SVM::C_SVC);     //SVM
	mode2->setKernel(SVM::LINEAR);  //核函數，這里使用線性核
	Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
	cout << "SVM: start train ..." << endl;
	mode2->trainAuto(tData);
	cout << "SVM: train success ..." << endl;
	mode2->save("svm123.xml");
}
void svm_pridect(Ptr<SVM> &mode2, Mat test) //利用訓練好的SVM預測
{
	Mat result;
	float rst = mode2->predict(test, result);
	for (auto i = 0; i < result.rows; i++)
	{
		cout << result.at<float>(i, 0);
	}
}
float sumMatValue(const Mat & image) // 計算影象中畫素灰度值總和
{
	float sumValue = 0;
	int r = image.rows;
	int c = image.cols;
	if (image.isContinuous()) //矩陣是否是連續的
	{
		c = r * c;
		r = 1;
	}
	for (int i = 0; i < r; i++)
	{
		const uchar *linePtr = image.ptr<uchar>(i);//尋訪全像素 與at相同
		for (int j = 0; j < c; j++)
		{
			sumValue += linePtr[j];
		}
	}
	return sumValue;
}
void calcGradientFeat(Mat & imgSrc, vector<float> & feat) //辨識字符歸一化
{
	Mat image;

	cvtColor(imgSrc, image, CV_BGR2GRAY);
	resize(image, image, Size(8, 16));
	float mask[3][3] = { { 1,2,1 },{ 0,0,0 },{ -1,-2,-1 } }; // 計算x方向和y方向上的濾波
	Mat y_mask = Mat(3, 3, CV_32F, mask) / 8;
	Mat x_mask = y_mask.t(); // 轉置
	Mat sobelX, sobelY;


	filter2D(image, sobelX, CV_32F, x_mask);//影像掃描
	filter2D(image, sobelY, CV_32F, y_mask);
	sobelX = abs(sobelX);//返回絕對值
	sobelY = abs(sobelY);


	float totleValueX = sumMatValue(sobelX);
	float totleValueY = sumMatValue(sobelY);
	// 將影象劃分為4*2共8個格子，計算每個格子裡灰度值總和的百分比
	for (int i = 0; i < image.rows; i = i + 4)
	{
		for (int j = 0; j < image.cols; j = j + 4)
		{
			Mat subImageX = sobelX(Rect(j, i, 4, 4));
			feat.push_back(sumMatValue(subImageX) / totleValueX);
			Mat subImageY = sobelY(Rect(j, i, 4, 4));
			feat.push_back(sumMatValue(subImageY) / totleValueY);
		}
	}


	Mat img2;
	resize(image, img2, Size(4, 8));
	int r = img2.rows;
	int c = img2.cols;
	if (img2.isContinuous()) {
		c = r * c;
		r = 1;
	}
	for (int i = 0; i < r; i++) {
		const uchar *linePtr = img2.ptr<uchar>(i);
		for (int j = 0; j < c; j++) {
			feat.push_back(linePtr[j]);
		}
	}
	//cout<<sobelX<<endl;
	//cout<<sobelY<<endl;
	//cout<< x_mask<<endl;
	//cout<<img2<<endl;
	/*for(int i=0; i<feat[num].size(); i++)
	{
	cout<<feat[i]<<endl;
	}*/
	//imshow("cat", img2);
	//cout<<"sumValue ="<<sumMatValue(image)<<endl;
}
Ptr<StatModel> buildMLPClassifier(Mat & input, Mat & output) //建立訓練檔
{
	Ptr<ANN_MLP> model;
	//train classifier;
	int layer_sz[] = { input.cols, 100 , output.cols };
	int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0])); //3
	/*cout << "input.cols = " << input.cols << endl;
	cout << "output.cols = " << output.cols << endl;*/
	Mat layer_sizes(1, nlayers, CV_32S, layer_sz); //[48 100 10]
	int method;
	double method_param;
	int max_iter;
	if (1)
	{
		method = ANN_MLP::BACKPROP;
		method_param = 0.01;
		max_iter = 100;
	}
	else
	{
		method = ANN_MLP::RPROP;
		method_param = 0.1;
		max_iter = 1000;
	}
	Ptr<TrainData> tData = TrainData::create(input, ROW_SAMPLE, output);
	model = ANN_MLP::create();
	cout << "create success" << endl;
	model->setLayerSizes(layer_sizes);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, max_iter, FLT_EPSILON));
	//setIterCondition(max_iter, 0);
	model->setTrainMethod(method, method_param);
	cout << "train data in process ...." << endl;
	model->train(tData);
	cout << "train success" << endl;
	model->save("mlp1.xml");
	return model;
}
Ptr<StatModel> loadMLPClassifiler() //讀取訓練檔
{
	Ptr<ANN_MLP> model = Algorithm::load<ANN_MLP>("mlp1.xml");
	return model;
}
int main()
{
	mysql_autocommit(&my_connection, 1);
	int svmnum = 0;
	Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2(200, 36.0, false); // 200 36 false
	pMOG2->setVarThreshold(300);
	string s;
	int img = 0, a = 1, b = 0, c = 0;	//變數img=原影像幀數 a=限制影像存取數 b=定位圖名稱
										//int img = 1;
										//string URL= "rtsp://192.168.43.25:8080/h264_ulaw.sdp";
	VideoCapture video("1111.mp4"); //讀取影像 20190515_124026 20190501_101514 20190501_101358 20190501_101627
	if (!video.isOpened())
	{
		return -1;
	}
	Size videoSize = Size((int)video.get(CV_CAP_PROP_FRAME_WIDTH), (int)video.get(CV_CAP_PROP_FRAME_HEIGHT));
	//namedWindow("video demo", CV_WINDOW_AUTOSIZE);
	Mat videoFrame;
	char filename[100];
	char fgmask[100];
	while (true)
	{
		if (!video.isOpened()) { exit(EXIT_FAILURE); }
		if (!video.read(frame)) { exit(EXIT_FAILURE); }           // 取前景
		pMOG2->apply(frame, fgMaskMOG2);                            // 取前景掩模
		stringstream ss;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20), cv::Scalar(255, 255, 255), -1);
		ss << video.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		imshow("Frame", frame);
		imshow("FG Mask MOG2", fgMaskMOG2);
		waitKey(30);
		char carnumber[10] = { 0 };
		video >> videoFrame;
		if (videoFrame.empty())
		{
			printf("break");
			break;
		}
		img++;
		resize(videoFrame, videoFrame, Size(1280, 720));//960 540 //1280 720 //1.77777*9
		imshow("video demo", videoFrame);
		if ((img % a) == 0 && !videoFrame.empty())
		{
			//存取圖片
			sprintf_s(filename, "picture/存取原圖/original_%d.jpg", img / a);
			imwrite(filename, videoFrame);
			resize(fgMaskMOG2, fgMaskMOG2, Size(1280, 720));
			sprintf_s(fgmask, "picture/存取動態圖/original_%d.jpg", img / a);
			imwrite(fgmask, fgMaskMOG2);
			Mat mog2src = imread(fgmask, CV_LOAD_IMAGE_GRAYSCALE);
			Mat src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
			//imshow("threshold 未二質化前", mog2src);
			//imshow("threshold 後", mog2src);
			//char car[100] = "1.png";
			//Mat src = imread(car, CV_LOAD_IMAGE_GRAYSCALE);
			Mat dst, dst1, dst2, dst3, dst4; //二值邊緣、濾波圖
			Mat lableImg;
			//medianBlur(src, src, 3); //中值濾波
			medianBlur(mog2src, dst, 3);
			//cout << dst << endl;
			//Canny(src, dst, 500, 250, 3);//邊緣化 500 250 300 150
			//threshold(dst, dst, 200, 255, CV_THRESH_BINARY);
			Canny(dst, dst, 500, 250, 3);
			threshold(dst, dst, 200, 255, CV_THRESH_BINARY);
			//imshow("threshold", dst);

			Mat dilate_image, erode_image; //侵蝕和膨脹
			Mat elementX = getStructuringElement(MORPH_RECT, Size(25, 1));
			Mat elementY = getStructuringElement(MORPH_RECT, Size(1, 13));//1 10 1 19
			Point point(-1, -1);
			//自定義 行 X Y 方向的膨脹腐蝕
			dilate(dst, dilate_image, elementX, point, 2);
			imwrite("dilate_image.jpg", dilate_image);
			erode(dilate_image, erode_image, elementX, point, 4);
			imwrite("erode_image.jpg", erode_image);
			dilate(erode_image, dilate_image, elementX, point, 2);
			imwrite("dilate_image1.jpg", dilate_image);
			erode(dilate_image, erode_image, elementY, point, 1);
			imwrite("erode_image1.jpg", erode_image);
			dilate(erode_image, dilate_image, elementY, point, 2);//2
			imwrite("dilate_image2.jpg", dilate_image);
			//噪音處理
			//平滑處理 中值濾波
			medianBlur(dilate_image, dst1, 15);
			medianBlur(dst1, dst1, 15);
			imshow("定位前影像處理 濾波", dst1);
			imwrite("定位前影像處理 濾波.jpg", dst1);
			waitKey(10);

			//SVM訓練
			vector<Mat>image;//設置兩類圖片的儲存，訓練數據
			string s1 = "C:/Users/user/source/repos/Project1/Project1/svm/正樣本/總集/test.txt";//個人路徑
			string sm, sa;
			string s2 = "C:/Users/user/source/repos/Project1/Project1/svm/負樣本/總集/test1.txt";
			ifstream fin1(s1);
			ifstream fin2(s2);
			while (getline(fin2, sa))
			{
				if (!fin2.eof())
				{
					//cout << "sa=" << sa << endl; //負樣本圖片讀取路徑顯示
					Mat m1 = imread(sa, 0);
					image.push_back(m1);  //連續放入Mat容器中
				}
				else
					break;
			}
			while (getline(fin1, sm))
			{
				if (!fin1.eof())
				{
					//cout << "sm=" << sm << endl; //正樣本圖片讀取路徑顯示
					Mat m1 = imread(sm, 0);
					image.push_back(m1);  //把圖片連續放入Mat容器中
				}
				else
					break;
			}
			//設置標籤數據，兩類，0或1
			int lable[172];//a=35 m=29 樣本數 
			for (int i = 0; i < 107; i++) //負樣本 sa.jpg+1 = 106 個
				lable[i] = 0;
			for (int j = 107; j < 172; j++) //正樣本  test+1-(sa.jpg +1) = 40個 66
				lable[j] = 1;
			Mat trainimage;
			transform(image, trainimage);
			Mat lableMat(172, 1, CV_32SC1, lable); //145 test+1 */
			Ptr<SVM> mode2 = SVM::create();
			//svm_train(mode2, trainimage, lableMat); //svm訓練
			dst2 = dst1.clone();
			vector<vector<Point>> contours;
			findContours(dst2, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


			//找出圖片綸廓
			drawContours(dst2, contours, -1, Scalar(255), 1);
			Mat roi_image;
			//vector<Point> rectPoint;
			for (int i = 0; i < contours.size(); i++)
			{
				Rect r = boundingRect(Mat(contours[i]));
				//RotatedRect r = minAreaRect(Mat(contours[i]));
				cout << "contours " << i << " ?height = " << r.height << " ?width = " << r.width << "rate = " << ((float)r.width / r.height) << endl;
				if ((float)r.width / r.height >= 2.0 && (float)r.width / r.height <= 3.6) //2.0 3.6
				{
					cout << "r.x = " << r.x << " r.y = " << r.y << endl;
					/*if (r.x < 1264 && r.y < 704 && r.x > 16 && r.y > 16)
					{
					r.x = r.x + 15;
					r.y = r.y - 15;
					}*/
					rectangle(dst3, r, Scalar(0, 0, 255), 2);
					imwrite("contour_image.jpg", dst3);
					/*Point p1, p2, p3, p4;
					p1.x = r.x;
					p1.y = r.y;
					p2.x = r.x + r.width;
					p2.x = r.y;
					p3.x = r.x + r.width;
					p3.y = r.y + r.height;
					p4.x = r.x;
					p4.y = r.y + r.height;
					rectPoint.push_back(p1);
					rectPoint.push_back(p2);
					rectPoint.push_back(p3);
					rectPoint.push_back(p4);*/
					/*for (int j = 0; j < contours[i].size(); j++)
					{
					cout << "point = " << contours[i][j] << endl;
					}*/
					char filename2[100];
					vector<Mat> svmimage;
					Mat svmtest;
					Mat svmlarge_image;
					svmimage.push_back(mog2src(r));
					sprintf_s(filename2, "picture/車牌定位疑似圖/original_%d.jpg", ++svmnum);
					transform(svmimage, svmlarge_image);//傳換成svm需要的模式
					Ptr<SVM> mode3 = SVM::load("svm123.xml");
					float svmresponse = mode3->predict(svmlarge_image, svmtest);
					for (auto i = 0; i < svmtest.rows; i++)
					{
						if (svmtest.at<float>(i, 0) == 1)
						{
							//cout << "第" << svmnum << "張車牌定位判斷" << endl;
							//cout << "svmtest.at<float> = " << svmtest.at<float>(i, 0) << endl; //svm預測 1為車牌 0非車牌
							//cout << "車牌" << endl;
							imwrite(filename2, mog2src(r));
							roi_image = mog2src(r);
						}
						if (svmtest.at<float>(i, 0) == 0)
						{
							//cout << "第" << svmnum << "張車牌定位判斷" << endl;
							//cout << "非車牌" << endl;
							//cout << "svmtest.at<float> = " << svmtest.at<float>(i, 0) << endl; //svm預測 1為車牌 0非車牌
							imwrite(filename2, mog2src(r));
							//roi_image = src(r);
						}
					}
					//svm_pridect(mode2, svmlarge_image);
					//cout << "第" << svmnum << "張車牌定位判斷" << endl;
					//imwrite(filename2, src(r));
					//roi_image = mog2src(r);
				}
			}
			if (!roi_image.empty()) //字元分割
			{
				Mat large_image;
				char filename3[100];
				int col = roi_image.cols, row = roi_image.rows;
				resize(roi_image, large_image, Size(300, 300 * row / col));
				//imshow("車牌定位", large_image);
				sprintf_s(filename3, "picture/車牌定位/original_%d.jpg", img / a);
				imwrite(filename3, large_image);
				waitKey(10);
				//中值濾波
				//Candy 邊緣化
				Mat candy_roi_image;
				Canny(large_image, candy_roi_image, 400, 120, 3);//400 200
				imwrite("candy_roi_image.jpg", candy_roi_image);
				imshow("邊緣", candy_roi_image);
				Mat roi_contours_image;
				vector<vector<Point>> roi_contours;
				roi_contours_image = candy_roi_image.clone();
				findContours(roi_contours_image, roi_contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
				vector<Point> roi_rectPoint;
				for (int i = 0; i < roi_contours.size(); i++) //找出邊界
				{
					Rect r = boundingRect(Mat(roi_contours[i]));
					//RotatedRect r = minAreaRect(Mat(contours[i]));
					//cout << "contours " << i << " height = " << r.height << " width = " << r.width << "rate = " << ((float)r.width / r.height) << endl;
					//cout << "r.x=" << r.x << "r.y=" << r.y << endl;
					//rectangle(large_image, r, Scalar(0, 0, 255), 1);
					/*Point p1, p2, p3, p4;

					p1.x = r.x;
					p1.y = r.y;
					p2.x = r.x + r.width;
					p2.x = r.y;
					p3.x = r.x + r.width;
					p3.y = r.y + r.height;
					p4.x = r.x;
					p4.y = r.y + r.height;
					roi_rectPoint.push_back(p1);
					roi_rectPoint.push_back(p2);
					roi_rectPoint.push_back(p3);
					roi_rectPoint.push_back(p4);*/
				}
				//判斷字符水平位置
				int roi_col = candy_roi_image.cols, roi_row = candy_roi_image.rows, position1[50], position2[50], roi_width[50];
				uchar pix;
				//cout << candy_roi_image << endl;
				int pixrow[10000];
				int num = 0;
				//去除多餘小汙漬 或多餘訊息
				//若一整列中有像素則為1 無則整列0
				for (int i = 0; i < roi_col - 1; i++)
				{
					for (int j = 0; j < roi_row - 1; j++)
					{
						pix = candy_roi_image.at<uchar>(j, i); //找出所有像素點
						pixrow[i] = 0;
						if (pix > 0)
						{
							pixrow[i] = 1;
							break;
						}
					}
				}
				for (int i = 2; i < roi_col - 1 - 2; i++)  //roi_col=340
				{
					if ((pixrow[i - 1] + pixrow[i - 2] + pixrow[i + 1] + pixrow[i + 2]) >= 3) //4列內有3列有像素 則視為目標列有像素
					{
						pixrow[i] = 1;
					}
					else if ((pixrow[i - 1] + pixrow[i - 2] + pixrow[i + 1] + pixrow[i + 2]) <= 1) //4列內有1列有像素 則視為目標列沒有像素
					{
						pixrow[i] = 0;
					}
				}
				int count = 0;
				bool flage = false;
				for (int i = 0; i < roi_col - 1; i++) //i=50,i=298 count=1 寬
				{
					pix = pixrow[i];
					if (pix == 1 && !flage)
					{
						flage = true;
						position1[count] = i;
						continue;
					}
					if (pix == 0 && flage)
					{
						flage = false;
						position2[count] = i;
						count++;
					}
					if (i == (roi_col - 2) && flage)
					{
						flage = false;
						position2[count] = i;
						count++;
					}
				}
				//所有字符寬度
				for (int n = 0; n < count; n++) {
					//cout << " position1 = " << position1[n] << " position2 = " << position2[n] << "distance =" << (position2[n] - position1[n]) << endl;
					roi_width[n] = position2[n] - position1[n];
				}
				//去最大值，最小值
				int max = roi_width[0], max_index = 0;
				int min = roi_width[0], min_index = 0;
				for (int n = 1; n < count; n++)
				{
					if (max < roi_width[n])
					{
						max = roi_width[n];
						max_index = n;
					}
					if (min > roi_width[n])
					{
						min = roi_width[n];
						min_index = n;
					}
				}
				int index = 0;
				int sum = 0;
				int new_roi_width[50];
				for (int i = 0; i < count; i++) //i=0 執行if內容 找出新寬度
				{
					if (i == min_index || i == max_index)
					{
						sum++;
						new_roi_width[count - sum] = 0;
					}
					else
					{
						new_roi_width[index] = roi_width[i];
						index++;
					}
				}
				//cout << "count = " << count << endl;
				int avgre = (int)((roi_width[max_index] + roi_width[min_index]) / 2.0); //最大值最小值平均數
				//cout << "avrge=" << avgre << endl;
				int licenseX[10] = { 0 }, licenseW[10] = { 0 }, licenseNum = 0;
				int countX = 0;
				for (int i = 0; i < count; i++) //i=0 count=6
				{
					//cout << "roi_width[i] = " << roi_width[i] << endl;
					if (roi_width[i] >= roi_width[min_index] && roi_width[i] <= roi_width[max_index] && roi_width[i] >= 10 && roi_width[i] <= 55) //roi_width>32 && roi_width<48  roi_width[i]=12 40 42 14 42 38
					{
						licenseX[licenseNum] = position1[i];
						licenseW[licenseNum] = roi_width[i];
						licenseNum++;
						//cout << "licenseX = " << licenseX[i] << " roi_width =" << roi_width[i] << endl;
						continue;
					}
					if (roi_width[i] >(avgre * 2 - 10) && roi_width[i] < (avgre * 2 + 10) && roi_width[i] >= 10 && roi_width[i] <= 55) //roi_width>70 && roi_width<90
					{
						licenseX[licenseNum] = position1[i];
						licenseW[licenseNum] = roi_width[i];
						licenseNum++;
						//cout << "licenseX = " << licenseX[i] << " roi_width =" << roi_width[i] << endl;
					}
				}
				//判斷字符垂直位置
				int licenseY[10] = { 0 }, licenseH[10] = { 0 };
				int position3[10], position4[10];
				//確認1 的像素
				int countYY = 0;
				int pixcol[1000], row_height[10];
				for (int temp = 0; temp < licenseNum; temp++) //若temp=0 licenseNum=0 不執行
				{
					for (int i = 0; i < roi_row - 1; i++)
					{
						for (int j = licenseX[temp]; j < (licenseX[temp] + licenseW[temp]); j++)
						{
							pix = candy_roi_image.at<uchar>(i, j);
							pixcol[i] = 0;
							if (pix > 0)
							{
								pixcol[i] = 1;
								break;
							}
						}
					}
					//執行數組濾波，減少突變概率 垂直
					for (int i = 2; i < roi_row - 1 - 2; i++)
					{
						if ((pixcol[i - 1] + pixcol[i - 2] + pixcol[i + 1] + pixcol[i + 2]) >= 3)
						{
							pixcol[i] = 1;
						}
						else if ((pixcol[i - 1] + pixcol[i - 2] + pixcol[i + 1] + pixcol[i + 2]) <= 1)
						{
							pixcol[i] = 0;
						}
					}
					//確認字符位置
					int countY = 0;
					bool flage2 = false;
					int min = 0, max = 0, count1 = 0;
					for (int i = 0; i < roi_row - 1; i++) //i=129 roi_row=131
					{
						pix = pixcol[i]; //1~10=0 11~129=1
						if (pix == 1 && !flage2) //0~12 29~52 //i=10
						{
							flage2 = true;
							position3[countY] = i;
							if (i > min && count1 < 1)
								min = i;
							count1++;
							continue;
						}
						if (pix == 0 && flage2) //12~29 52~85 //i=10~129
						{
							flage2 = false;
							position4[countY] = i;
							if (i > max)
								max = i;
							countY++;
						}
					}
					//所有字符高度
					for (int n = 0; n < countY; n++)
					{
						//cout << " position3 = " << position3[n] << " position4 = " << position4[n] << "distance =" << (position4[n] - position3[n]) << endl;
						//cout << " max = " << max << " min = " << min << endl;
						if (position4[n] - position3[n] <= max - min && position4[n] - position3[n] >= 20) // 0 6 =6   23 98 =75 // 98 0 
						{
							row_height[countYY] = position4[n] - position3[n];
							licenseY[countYY] = position3[n];
						}
						else
						{
							row_height[countYY] = max - min;
							licenseY[countYY] = min;
						}
						licenseH[countYY] = row_height[countYY];
					}
					countYY++;
				}
				//截取字符
				Mat licenseN = Mat(Scalar(0));
				char filename1[100];
				int n = 0;
				//cout << "countYY = " << countYY << endl;
				if (countYY >= 7)
					for (n = 0; n < countYY; n++)
					{
						int cutnum = c;
						Rect rect(licenseX[n], licenseY[n], licenseW[n], licenseH[n]);
						//cout << "licenseX = " << licenseX[i] << " licenseY=" << licenseY[i] << " licenseW=" << licenseW[i] << " licenseH=" << licenseH[i] << endl;
						ostringstream oss;
						oss << "licenseN" << n << ".png";
						//cout << "第" << c << "張車牌分割" << endl;
						sprintf_s(filename1, "picture/車牌分割/%d.png", c++);
						imwrite(filename1, large_image(rect));
						if (countYY >= 7) //車牌辨識
						{
							Mat image, image1;
							vector<float>feats;
							vector<float>test, test1;
							string path = "C:/Users/user/source/repos/Project1/Project1/train/";
							string cut = "C:/Users/user/source/repos/Project1/Project1/picture/車牌分割/";
							int numbers = 0;
							int classfilternum = 35;
							int modlenum = 30;
							for (int i = 0; i < classfilternum; i++)
							{
								for (int j = 0; j < modlenum; j++)
								{
									ostringstream oss;
									oss << path << i << "/" << j << ".png";
									//cout << oss.str() << endl;
									image = imread(oss.str());
									calcGradientFeat(image, feats);//歸一化
									numbers++;
									if (i == 10 && j == 11)
									{
										ostringstream oss;
										oss << cut << cutnum << ".png";
										//oss << path << i << "/" << (j+1) << ".png";
										//cout << oss.str() << endl;
										image1 = imread(oss.str(), 1);
										calcGradientFeat(image1, test);
									}
								}
							}
							Mat input, output;
							input = Mat(classfilternum*modlenum, 48, CV_32F);
							output = Mat(classfilternum*modlenum, classfilternum, CV_32F, Scalar(0));
							int row = input.rows;
							int col = input.cols;
							if (input.isContinuous())
							{
								col = row * col;
								row = 1;
							}
							for (int i = 0; i < row; i++)
							{
								float *linePtr = input.ptr<float>(i);
								for (int j = 0; j < col; j++)
								{
									linePtr[j] = feats[col*i + j];
								}
							}
							for (int i = 0; i < output.rows; i++)
							{
								float *lineoutput = output.ptr<float>(i);
								lineoutput[i / modlenum] = 1;
							}
							//if(carnumber
							//Ptr<StatModel> model = buildMLPClassifier(input, output); //建置訓練檔
							Ptr<StatModel> model = loadMLPClassifiler(); //讀取訓練檔
							float response = model->predict(test, test1);//車牌號碼
							//cout << "response = " << response << endl;
							if (response < 10)//1~10
							{
								char Number = response + 48;
								//cout << "response = " << Number << endl;
								carnumber[n] = Number;
							}
							else if (response > 9 && response <= 17)//A~H
							{
								char English = response + 55;
								//cout << "response = " << English << endl;
								carnumber[n] = English;
							}
							else if (response > 17 && response <= 22)//J~N
							{
								char English2 = response + 56;
								//cout << "response = " << English2 << endl;
								carnumber[n] = English2;
							}
							else if (response > 22 && response <= 33)//N~Z
							{
								char English3 = response + 57;
								//cout << "response = " << English3 << endl;
								carnumber[n] = English3;
							}
							else if (response == 34)
							{
								//cout << "response =" << "-" << endl;
								carnumber[n] = '-';
							}
							/*for (int i = 0; i < test1.size(); i++)
							{
							//cout << "test1 = " << test1[i] << i << endl; //概率
							}*/
							//cout << "carnumber測試 = " << carnumber << endl;
						}
						//mysql 程式					
					}
				int syow = 0;
				for (int s = 0; s <= 9; s++)
				{
					if (carnumber[s] == '-')
						syow++;
				}
				if (n == countYY && countYY != 0 && n != 0 && syow < 2)
				{
					cout << "carnumber = " << carnumber << endl;
					mysql_init(&my_connection);
					mysql_real_connect(&my_connection, "127.0.0.1", "test", "cartest", "smart traffic light", 0, NULL, 0);
					char str[100] = "INSERT INTO `car` (`time`, `carnumber`) VALUES(CURRENT_TIMESTAMP, '";
					strcat_s(str, 100, carnumber);
					strcat_s(str, 100, "')");
					mysql_query(&my_connection, str);
				}
			}
		}
	}
}
