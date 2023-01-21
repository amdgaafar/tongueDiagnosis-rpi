#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

const int MAX_KERNEL_LENGTH = 16; // Blur strength
const int CLUSTERS = 4;

Mat segBCM(const Mat &inputImage, Mat &outputMask);
Mat coatingRemoval(const Mat &segmentedImage, Mat &outputMask);
Mat clusters(Mat noCoatingImage, int K);

int main(int argc, char *argv[])
{
    int imageArgumentIdx = 1;

    // Reads the image.
    Mat imagereceive_data = imread(argv[imageArgumentIdx]);
    if (!image.data)
    {
        cout << "Image not found or empty" << endl;
        return 1;
    }
    cout << "Original Image: "
         << "Height: " << image.rows << ", Width: " << image.cols << ", Channels: " << image.channels() << endl;

    Mat tongueMask, noCoatingMask, detectedTongue, noCoating; // Images

    // Tongue Segmentation
    segBCM(image, tongueMask);
    image.copyTo(detectedTongue, tongueMask);

    // Remove coating and substance from the tongue
    coatingRemoval(detectedTongue, noCoatingMask);
    detectedTongue.copyTo(noCoating, noCoatingMask);

    // k-means (clustring the image)
    clusters(noCoating, CLUSTERS);

    imwrite("output_images/im(segmented)-rpi.png", detectedTongue);
    imwrite("output_images/im(finaloutput)-rpi.bmp", noCoating);

    return 0;
}

Mat segBCM(const Mat &inputImage, Mat &outputMask)
{
    // Applys Gaussian smoothing on the original image
    cv::Mat imageBlur;
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 4)
    {
        cv::GaussianBlur(inputImage, imageBlur, cv::Size(i, i), 0, 0);
    }

    // Converts RGB to HSV space conversion.
    Mat hsvImg;
    cvtColor(imageBlur, hsvImg, cv::COLOR_BGR2HSV);

    // Splits the HSV image to get the brightness component.
    vector<Mat> components;
    split(hsvImg, components);

    // Gets the mean and the standard deviation of the V component.
    Scalar meanImgV, stdDevImgV;
    meanStdDev(components[2], meanImgV, stdDevImgV);

    // Calculates the eps of the BCM
    double eps{0};
    double vLower = meanImgV[0] - stdDevImgV[0];
    double vMin{0};
    minMaxLoc(components[2], &vMin);

    if (vLower < (2 * stdDevImgV[0]))
    {
        eps = meanImgV[0];
    }
    else if (vLower > (2 * stdDevImgV[0]))
    {
        eps = vMin;
    }
    else
        eps = meanImgV[0] * meanImgV[0];

    double vUpper = meanImgV[0] + (eps * (stdDevImgV[0] / 255));

    // Thresholds the image based on the BCM
    Mat mask1, mask2;

    // V masking
    threshold(components[2], mask1, vLower, 255, THRESH_BINARY); // Select after the vLower
    threshold(components[2], mask2, vUpper, 255, THRESH_BINARY); // Select after the vUpper

    // Applying the Erode morphological operation and deleting the small areas
    cv::Mat box(1, 1, CV_8U, cv::Scalar(1));
    cv::morphologyEx(mask2, mask2, cv::MORPH_ERODE, box, Point(-1, -1), 3); // ERODE 3 times

    // Finds the contours from the maskAdapV
    std::vector<std::vector<cv::Point>> contoursMask2;
    std::vector<cv::Vec4i> hierarchyMask2;
    cv::Mat imageContoursMask2(inputImage.size(), CV_8UC1, cv::Scalar(0));
    cv::findContours(mask2, contoursMask2, hierarchyMask2, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Finds the largest area from the contours
    int idx = 0, largestComp = 0, maxArea = 0;
    for (; idx >= 0; idx = hierarchyMask2[idx][0])
    {
        const std::vector<cv::Point> &c = contoursMask2[idx];
        double area = fabs(contourArea(cv::Mat(c)));
        if (area > maxArea)
        {
            maxArea = area;
            largestComp = idx;
        }
    }

    cv::Scalar whiteColour(255, 255, 255);
    drawContours(imageContoursMask2, contoursMask2, largestComp, whiteColour, cv::FILLED, cv::LINE_8, hierarchyMask2);
    outputMask = imageContoursMask2;
    return outputMask;
}

Mat coatingRemoval(const Mat &segmentedImage, Mat &outputMask)
{
    // Applys Gaussian smoothing on the original image
    cv::Mat segmentedImageBlur;
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 4)
    {
        cv::GaussianBlur(segmentedImage, segmentedImageBlur, cv::Size(i, i), 0, 0);
    }

    // Converts RGB to HSV space conversion.
    Mat hsvSegImg;
    cvtColor(segmentedImage, hsvSegImg, cv::COLOR_BGR2HSV);

    // Splits the HSV image
    vector<Mat> components;
    split(hsvSegImg, components);

    // Algorithm III and IV (Coating Removal)
    Mat w_t1 = ((components[2] - (components[1] + components[0])) / 3);

    // Converts the image into binary image
    outputMask = ~(w_t1 * 255);
    return outputMask;
}

Mat clusters(Mat inputImage, int K)
{
    cvtColor(inputImage, inputImage, cv::COLOR_BGR2Lab);
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int channels = inputImage.channels();

    Mat samples((rows * cols), channels, CV_32F); // Reshapped image, it should be of float32 data type, and each feature(channel) should be put in a single column.

    // Copy the image to samples (Reshaping the image)
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
            for (int z = 0; z < channels; z++)
                samples.at<float>(y + x * rows, z) = inputImage.at<Vec3b>(y, x)[z];

    cout << "samples image (Reshaped image): "
         << "Rows: " << samples.rows << ", Cols: " << samples.cols << ", Channles: " << samples.channels() << endl;

    Mat labels, centers;
    vector<Vec3b> clusteredColours;

    double eps = 0.0001;
    int max_iter = 1000;
    int attempts = 5;

    cv::kmeans(samples, K, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, max_iter, eps), attempts, KMEANS_PP_CENTERS, centers);

    Mat clusteredImage(inputImage.size(), inputImage.type(), Scalar(0, 0, 0));
    Mat cluster0(inputImage.size(), inputImage.type(), Scalar(0, 0, 0));
    Mat cluster1(inputImage.size(), inputImage.type(), Scalar(0, 0, 0));
    Mat cluster2(inputImage.size(), inputImage.type(), Scalar(0, 0, 0));
    Mat cluster3(inputImage.size(), inputImage.type(), Scalar(0, 0, 0));
    // cvtColor(clusteredImage, clusteredImage, cv::COLOR_Lab2BGR);
    // cvtColor(cluster0, cluster0, cv::COLOR_Lab2BGR);
    // cvtColor(cluster1, cluster1, cv::COLOR_Lab2BGR);
    // cvtColor(cluster2, cluster2, cv::COLOR_Lab2BGR);
    // cvtColor(cluster3, cluster3, cv::COLOR_Lab2BGR);
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            int cluster_idx = labels.at<int>(y + x * rows, 0); // Cluster inedx

            // Creates the clustered image
            for (int i = 0; i < channels; i++)
            {
                float pixValue = centers.at<float>(cluster_idx, i);
                clusteredImage.at<Vec3b>(y, x)[i] = pixValue;
            }

            // Finds the colours used for the clusters
            Vec3b pix = clusteredImage.at<Vec3b>(y, x);
            if (std::find(clusteredColours.begin(), clusteredColours.end(), pix) == clusteredColours.end())
            {
                clusteredColours.push_back(pix);
            }
        }
    }

    // Print the colours
    cout << "----clustered colours----" << endl;
    for (int i = 0; i < K; i++)
    {
        cout << clusteredColours[i] << endl;
    }

    // Create one image for individual clusters
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            Vec3b pix = clusteredImage.at<Vec3b>(y, x);

            if (pix == clusteredColours[1])
            {
                cluster1.at<Vec3b>(y, x) = pix;
            }
            else if (pix == clusteredColours[2])
            {
                cluster2.at<Vec3b>(y, x) = pix;
            }
            else if (pix == clusteredColours[3])
            {
                cluster3.at<Vec3b>(y, x) = pix;
            }
            else
                cluster0.at<Vec3b>(y, x) = pix;
        }
    }

    imwrite("output_imgs/Clustered Image.png", clusteredImage);
    imwrite("output_imgs/Cluster 0 (Background).png", cluster0);
    imwrite("output_imgs/Cluster 1.png", cluster1);
    imwrite("output_imgs/Cluster 2.png", cluster2);
    imwrite("output_imgs/Cluster 3.png", cluster3);
    return clusteredImage;
}
