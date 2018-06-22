

#include "sift.h"
#include "image_io.h"
#include "image_tools.h"


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
	

	/*while (img->width() * img->height() > 6000000)
		img = image::rescale_half_size<uint8_t>(img);*/

	sift.set_image(img);
	sift.process();
	descr = sift.get_descriptors();

	std::sort(descr.begin(), descr.end(), compare_scale<Sift::Descriptor>);


 	return 0;
}