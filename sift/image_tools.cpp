#include <algorithm>

#include "image_tools.h"

namespace image
{
	FloatImage::Ptr
		byte_to_float_image(ByteImage::ConstPtr image)
	{
		if (image == nullptr)
			throw std::invalid_argument("Null image given");

		FloatImage::Ptr img = FloatImage::create();
		img->allocate(image->width(), image->height(), image->channels());
		for (int i = 0; i < image->get_value_amount(); ++i)
		{
			float value = (float)image->at(i) / 255.0f;
			img->at(i) = std::min(1.0f, std::max(0.0f, value));
		}
		return img;
	}





}