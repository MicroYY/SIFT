

#include "sift.h"
#include "image_io.h"
#include "image_tools.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

template <typename T>
bool
compare_scale(T const& descr1, T const& descr2)
{
	return descr1.scale > descr2.scale;
}


int main(int argc,char** argv)
{
	std::string image_path = argv[1];

	Sift::Descriptors descr;
	Sift::Config conf;
	Sift sift(conf);


	/*image::ImageBase::Ptr image = image::load_file(image_path);
	image::ByteImage::Ptr img = std::dynamic_pointer_cast<image::ByteImage>(image);*/
	image::ByteImage::Ptr img = image::load_file(image_path);
	image::save_jpg_file(img, "j1.jpg", 100);
	cv::Mat mat = cv::imread(image_path);
	cv::imwrite("j2.jpg", mat);
	while (img->width() * img->height() > 6000000)
	{
		img = image::rescale_half_size<uint8_t>(img);
		cv::resize(mat, mat, cv::Size(mat.cols / 2, mat.rows / 2), 0, 0);
	}


	sift.set_image(img);
	sift.process();
	descr = sift.get_descriptors();

	std::sort(descr.begin(), descr.end(), compare_scale<Sift::Descriptor>);

	
	//cv::resize(mat, mat, cv::Size(mat.cols / 2, mat.rows / 2), 0, 0);
	//cv::imwrite("1234.jpg", mat);
	int i = 0;
	for each (Sift::Descriptor var in descr)
	{
		i++;
		cv::circle(mat, cv::Point(var.x, var.y),2, cv::Scalar(255, 0, 0));
	}
	std::cout << i;
	cv::imshow("", mat);
	cv::imwrite(argv[2], mat);
	cv::waitKey();
 	return 0;
}