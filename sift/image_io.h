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


	/************************ ͼ������ͱ��� *************************/
	
	/*
		����ͼƬ
	*/
	ByteImage::Ptr
		load_file(std::string const& filename);


	/************************* PNG *************************/

	/*
		����PNG
		PNG��gray, gray-alpha, RGB, RGBA
	*/
	ByteImage::Ptr
		load_png_file(std::string const& filename);



	/************************* JPG *************************/

	/*
		����JPEG
		JPEG�� ��ͨ��gray����ͨ��RGB
	*/
	ByteImage::Ptr
		load_jpg_file(std::string const& filename, std::string* exif = nullptr);

	

}
