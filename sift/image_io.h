#pragma once

#include <string>

#include "image.h"


namespace image
{

	struct ImageHeaders
	{
		int width;
		int height;
		int channels;
		ImageType type;
	};


	/************************ 图像载入和保存 *************************/
	
	/*
		载入图片
	*/
	ByteImage::Ptr
		load_file(std::string const& filename);


	/************************* PNG *************************/

	/*
		载入PNG
		PNG：gray, gray-alpha, RGB, RGBA
	*/
	ByteImage::Ptr
		load_png_file(std::string const& filename);



	/************************* JPG *************************/

	/*
		载入JPEG
		JPEG： 单通道gray或三通道RGB
	*/
	ByteImage::Ptr
		load_jpg_file(std::string const& filename, std::string* exif = nullptr);

	

}
