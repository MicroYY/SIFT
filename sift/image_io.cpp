#include <algorithm>





#ifndef NO_PNG_SUPPORT
#include <png.h>
#endif // !NO_PNG_SUPPORT

#ifndef NO_JPG_SUPPORT
#if defined _WIN32
#include <Windows.h>
#endif
#include <jpeglib.h>
#endif // !NO_JPG_SUPPORT

#ifndef  NO_TIFF_SUPPORT
#include <tiff.h>
#include <tiffio.h>
#endif // ! NO_TIFF_SUPPORT

#include "algo.h"
#include "image_io.h"



namespace image
{
	
	ByteImage::Ptr
		load_file(std::string const& filename)
	{
		try
		{
#ifndef NO_PNG_SUPPORT
			try
			{
				return load_png_file(filename);
			}
			catch (const std::exception&)
			{

			}
#endif // !NO_PNG_SUPPORT

#ifndef NO_JPG_SUPPORT
			try
			{
				return load_jpg_file(filename);
			}
			catch (const std::exception&)
			{

			}
#endif // !NO_JPG_SUPPORT



		}
		catch (const std::exception&)
		{

		}
	}


#ifndef NO_PNG_SUPPORT

	void
		load_png_headers_intern(FILE* fp, ImageHeaders* headers,
			png_structp* png, png_infop* png_info)
	{
		/* Identify the PNG signature. */
		png_byte signature[8];
		if (std::fread(signature, 1, 8, fp) != 8)
		{
			std::fclose(fp);
			throw std::exception("PNG signature could not be read");
		}
		if (png_sig_cmp(signature, 0, 8) != 0)
		{
			std::fclose(fp);
			throw std::exception("PNG signature did not match");
		}

		/* Initialize PNG structures. */
		*png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
			nullptr, nullptr, nullptr);
		if (!*png)
		{
			std::fclose(fp);
			throw std::exception("Out of memory");
		}

		*png_info = png_create_info_struct(*png);
		if (!*png_info)
		{
			png_destroy_read_struct(png, nullptr, nullptr);
			std::fclose(fp);
			throw std::exception("Out of memory");
		}

		/* Init PNG file IO */
		png_init_io(*png, fp);
		png_set_sig_bytes(*png, 8);

		/* Read PNG header info. */
		png_read_info(*png, *png_info);

		headers->width = png_get_image_width(*png, *png_info);
		headers->height = png_get_image_height(*png, *png_info);
		headers->channels = png_get_channels(*png, *png_info);

		int const bit_depth = png_get_bit_depth(*png, *png_info);
		if (bit_depth <= 8)
			headers->type = IMAGE_TYPE_UINT8;
		else if (bit_depth == 16)
			headers->type = IMAGE_TYPE_UINT16;
		else
		{
			png_destroy_read_struct(png, png_info, nullptr);
			std::fclose(fp);
			throw std::exception("PNG with unknown bit depth");
		}
	}


	ByteImage::Ptr
		load_png_file(std::string const& filename)
	{
		FILE* fp = std::fopen(filename.c_str(), "rb");
		if (fp == nullptr)
			throw std::exception();

		//读头文件
		ImageHeaders headers;
		png_structp png = nullptr;
		png_infop png_info = nullptr;
		load_png_headers_intern(fp, &headers, &png, &png_info);

		//检查位深度是否有效
		int const bit_depth = png_get_bit_depth(png, png_info);
		if (bit_depth > 8)
		{
			png_destroy_read_struct(&png, &png_info, nullptr);
			std::fclose(fp);
			throw std::exception("PNG with more than 8 bit");
		}

		//
		int const color_type = png_get_color_type(png, png_info);
		if (color_type == PNG_COLOR_TYPE_PALETTE)
			png_set_palette_to_rgb(png);
		if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
			png_set_expand_gray_1_2_4_to_8(png);
		if (png_get_valid(png, png_info, PNG_INFO_tRNS))
			png_set_tRNS_to_alpha(png);

		/* Update the info struct to reflect the transformations. */
		png_read_update_info(png, png_info);

		/* Create image. */
		ByteImage::Ptr image = ByteImage::create();
		image->allocate(headers.width, headers.height, headers.channels);
		ByteImage::ImageData& data = image->get_data();

		/* Setup row pointers. */
		std::vector<png_bytep> row_pointers;
		row_pointers.resize(headers.height);
		for (int i = 0; i < headers.height; ++i)
			row_pointers[i] = &data[i * headers.width * headers.channels];

		/* Read the whole PNG in memory. */
		png_read_image(png, &row_pointers[0]);

		/* Clean up. */
		png_destroy_read_struct(&png, &png_info, nullptr);
		std::fclose(fp);

		return image;

	}


#endif // !NO_PNG_SUPPORT


#ifndef NO_JPG_SUPPORT

	void
		jpg_error_handler(j_common_ptr /*cinfo*/)
	{
		throw std::exception("JPEG format nor recognized");
	}


	void
		jpg_message_handler(j_common_ptr /*cinfo*/, int msg_level)
	{
		if (msg_level < 0)
			throw std::exception("JPEG data corrupt");
	}


	ByteImage::Ptr
		load_jpg_file(std::string const& filename, std::string* exif)
	{
		FILE* fp = std::fopen(filename.c_str(), "rb");
		if (fp == nullptr)
			throw std::exception();

		jpeg_decompress_struct cinfo;
		jpeg_error_mgr jerr;
		ByteImage::Ptr image;
		try
		{
			/* Setup error handler and JPEG reader. */
			cinfo.err = jpeg_std_error(&jerr);
			jerr.error_exit = &jpg_error_handler;
			jerr.emit_message = &jpg_message_handler;
			jpeg_create_decompress(&cinfo);
			jpeg_stdio_src(&cinfo, fp);

			if (exif)
			{
				/* Request APP1 marker to be saved (this is the EXIF data). */
				jpeg_save_markers(&cinfo, JPEG_APP0 + 1, 0xffff);
			}

			/* Read JPEG header. */
			int ret = jpeg_read_header(&cinfo, static_cast<boolean>(false));
			if (ret != JPEG_HEADER_OK)
				throw std::exception("JPEG header not recognized");

			/* Examine JPEG markers. */
			if (exif)
			{
				jpeg_saved_marker_ptr marker = cinfo.marker_list;
				if (marker != nullptr && marker->marker == JPEG_APP0 + 1
					&& marker->data_length > 6
					&& std::equal(marker->data, marker->data + 6, "Exif\0\0"))
				{
					char const* data = reinterpret_cast<char const*>(marker->data);
					exif->append(data, data + marker->data_length);
				}
			}

			if (cinfo.out_color_space != JCS_GRAYSCALE
				&& cinfo.out_color_space != JCS_RGB)
				throw std::exception("Invalid JPEG color space");

			/* Create image. */
			int const width = cinfo.image_width;
			int const height = cinfo.image_height;
			int const channels = (cinfo.out_color_space == JCS_RGB ? 3 : 1);
			image = ByteImage::create(width, height, channels);
			ByteImage::ImageData& data = image->get_data();

			/* Start decompression. */
			jpeg_start_decompress(&cinfo);

			unsigned char* data_ptr = &data[0];
			while (cinfo.output_scanline < cinfo.output_height)
			{
				jpeg_read_scanlines(&cinfo, &data_ptr, 1);
				data_ptr += channels * cinfo.output_width;
			}

			/* Shutdown JPEG decompression. */
			jpeg_finish_decompress(&cinfo);
			jpeg_destroy_decompress(&cinfo);
			std::fclose(fp);
		}
		catch (...)
		{
			jpeg_destroy_decompress(&cinfo);
			std::fclose(fp);
			throw;
		}

		return image;

	}


#endif // !NO_JPG_SUPPORT







}