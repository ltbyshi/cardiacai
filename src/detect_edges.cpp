#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace cv;

int main(int argc, char** argv)
{
    if(argc != 3){
        fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        exit(1);
    }
    String input_file(argv[1]);
    String output_file(argv[2]);

    Mat src, src_gray;
    Mat grad;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    int kernel_size = 3;

    /// Load an image
    src = imread(input_file);

    if (!src.data)
    {
        return -2;
    }

    GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
    cvtColor(src, src_gray, CV_RGB2GRAY);


    /////////////////////////// Sobel ////////////////////////////////////
    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    //Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
    Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    /// Total Gradient (approximate)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    //global 2
    int th = 27;
    Mat global;
    threshold(grad, global, th, 255, CV_THRESH_BINARY_INV);

    imwrite(output_file, global);

	return 0;
}