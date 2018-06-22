#pragma once

#include <cstdint>
#include <memory>
#include <vector>


namespace image {

	enum ImageType
	{
		IMAGE_TYPE_UNKNOWN,

		IMAGE_TYPE_UINT8,
		IMAGE_TYPE_UINT16,
		IMAGE_TYPE_UINT32,
		IMAGE_TYPE_UINT64,

		IMAGE_TYPE_SINT8,
		IMAGE_TYPE_SINT16,
		IMAGE_TYPE_SINT32,
		IMAGE_TYPE_SINT64,

		IMAGE_TYPE_FLOAT,
		IMAGE_TYPE_DOUBLE
	};

	/*
		图像基类，没有类型信息
		提供了图像尺寸、通道信息和数据访问的框架
	*/
	class ImageBase
	{
	public:
		typedef std::shared_ptr<ImageBase> Ptr;
		typedef std::shared_ptr<ImageBase const> ConstPtr;

	public:
		ImageBase();
		virtual ~ImageBase();

		virtual ImageBase::Ptr duplicate_base() const;

		int width() const;
		int height() const;
		int channels() const;

		/*
			width,height,channels存在0时返回false
		*/
		bool valid() const;

		bool reinterpret(int w, int h, int c);

		virtual std::size_t get_byte_size() const;

		virtual char const* get_byte_pointer() const;

		virtual char* get_byte_pointer();

		virtual ImageType get_type() const;

		virtual char const* get_type_string() const;


		static ImageType get_type_for_string(std::string const& type_string);

	protected:
		int w, h, c;

	};


	inline
		ImageBase::ImageBase()
		:w(0), h(0), c(0)
	{
	}

	inline
		ImageBase::~ImageBase()
	{
	}

	inline ImageBase::Ptr
		ImageBase::duplicate_base() const
	{
		return ImageBase::Ptr(new ImageBase(*this));
	}


	inline int
		ImageBase::width() const
	{
		return this->w;
	}

	inline int
		ImageBase::height() const
	{
		return this->h;
	}

	inline int
		ImageBase::channels() const
	{
		return this->c;
	}

	inline bool
		ImageBase::valid() const
	{
		return this->w && this->h && this->c;
	}

	inline bool
		ImageBase::reinterpret(int w, int h, int c)
	{
		if (w * h * c != this->w * this->h * this->c)
			return false;
		this->w = w;
		this->h = h;
		this->c = c;
		return true;
	}

	inline std::size_t
		ImageBase::get_byte_size() const
	{
		return 0;
	}

	inline char const*
		ImageBase::get_byte_pointer() const
	{
		return nullptr;
	}

	inline char*
		ImageBase::get_byte_pointer()
	{
		return nullptr;
	}

	inline ImageType
		ImageBase::get_type() const
	{
		return IMAGE_TYPE_UNKNOWN;
	}

	inline char const*
		ImageBase::get_type_string() const
	{
		return "unknown";
	}

	inline ImageType
		ImageBase::get_type_for_string(std::string const& type_string)
	{
		if (type_string == "sint8")
			return IMAGE_TYPE_SINT8;
		else if (type_string == "sint16")
			return IMAGE_TYPE_SINT16;
		else if (type_string == "sint32")
			return IMAGE_TYPE_SINT32;
		else if (type_string == "sint64")
			return IMAGE_TYPE_SINT64;
		else if (type_string == "uint8")
			return IMAGE_TYPE_UINT8;
		else if (type_string == "uint16")
			return IMAGE_TYPE_UINT16;
		else if (type_string == "uint32")
			return IMAGE_TYPE_UINT32;
		else if (type_string == "uint64")
			return IMAGE_TYPE_UINT64;
		else if (type_string == "float")
			return IMAGE_TYPE_FLOAT;
		else if (type_string == "double")
			return IMAGE_TYPE_DOUBLE;

		return IMAGE_TYPE_UNKNOWN;
	}


	template<typename T>
	class TypedImageBase : public ImageBase
	{
	public:
		typedef T ValueType;
		typedef std::shared_ptr<TypedImageBase<T>> Ptr;
		typedef std::shared_ptr<TypedImageBase<T> const> ConstPtr;
		typedef std::vector<T> ImageData;

	public:
		TypedImageBase();

		TypedImageBase(TypedImageBase<T> const& src);

		virtual ~TypedImageBase();

		virtual ImageBase::Ptr duplicate_base() const;

		void allocate(int width, int height, int channels);

		void resize(int width, int height, int channels);

		virtual void clear();

		void fill(T const& value);

		void swap(TypedImageBase<T>& other);

		virtual ImageType get_type() const;

		char const* get_type_string() const;

		ImageData const& get_data() const;

		ImageData& get_data();

		T const* get_data_pointer() const;

		T* get_data_pointer();

		T* begin();

		T const* begin() const;

		T* end();

		T const* end() const;

		int get_pixel_amount() const;

		int get_value_amount() const;

		std::size_t get_byte_size() const;

		char const* get_byte_pointer() const;

		char* get_byte_pointer();

	protected:
		//size = width * height * channels
		ImageData data;
	};

	template<typename T>
	inline
		TypedImageBase<T>::TypedImageBase()
	{
	}

	template<typename T>
	inline TypedImageBase<T>::TypedImageBase(TypedImageBase<T> const& src)
		:ImageBase(src), data(src.data)
	{
	}

	template<typename T>
	inline
		TypedImageBase<T>::~TypedImageBase()
	{
	}

	template<typename T>
	inline ImageBase::Ptr
		TypedImageBase<T>::duplicate_base() const
	{
		return ImageBase::Ptr(new TypedImageBase<T>(*this));
	}

	template<typename T>
	inline ImageType
		TypedImageBase<T>::get_type() const
	{
		return IMAGE_TYPE_UNKNOWN;
	}

	template<>
	inline ImageType
		TypedImageBase<int8_t>::get_type() const
	{
		return IMAGE_TYPE_SINT8;
	}

	template<>
	inline ImageType
		TypedImageBase<int16_t>::get_type() const
	{
		return IMAGE_TYPE_SINT16;
	}

	template<>
	inline ImageType
		TypedImageBase<int32_t>::get_type() const
	{
		return IMAGE_TYPE_SINT32;
	}

	template<>
	inline ImageType
		TypedImageBase<int64_t>::get_type() const
	{
		return IMAGE_TYPE_SINT64;
	}

	template<>
	inline ImageType
		TypedImageBase<uint8_t>::get_type() const
	{
		return IMAGE_TYPE_UINT8;
	}

	template<>
	inline ImageType
		TypedImageBase<uint16_t>::get_type() const
	{
		return IMAGE_TYPE_UINT16;
	}

	template<>
	inline ImageType
		TypedImageBase<uint32_t>::get_type() const
	{
		return IMAGE_TYPE_UINT32;
	}

	template<>
	inline ImageType
		TypedImageBase<uint64_t>::get_type() const
	{
		return IMAGE_TYPE_UINT64;
	}

	template<>
	inline ImageType
		TypedImageBase<float>::get_type() const
	{
		return IMAGE_TYPE_FLOAT;
	}

	template<>
	inline ImageType
		TypedImageBase<double>::get_type() const
	{
		return IMAGE_TYPE_DOUBLE;
	}

	template<typename T>
	inline char const*
		TypedImageBase<T>::get_type_string() const
	{
		//TODO
		return "unknown";
	}

	template<typename T>
	inline void
		TypedImageBase<T>::allocate(int width, int height, int channels)
	{
		this->clear();
		this->resize(width, height, channels);
	}

	template<typename T>
	inline void
		TypedImageBase<T>::resize(int width, int height, int channels)
	{
		this->w = width;
		this->h = height;
		this->c = channels;
		this->data.resize(width * height * channels);
	}

	template<typename T>
	inline void
		TypedImageBase<T>::clear()
	{
		this->w = 0;
		this->h = 0;
		this->c = 0;
		this->data.clear();
	}

	template<typename T>
	inline void
		TypedImageBase<T>::fill(T const& value)
	{
		std::fill(this->data.begin(), this->data.end(), value);
	}

	template<typename T>
	inline void
		TypedImageBase<T>::swap(TypedImageBase<T>& other)
	{
		std::swap(this->w, other.w);
		std::swap(this->h, other.h);
		std::swap(this->c, other.c);
		std::swap(this->data, other.data);
	}

	template<typename T>
	inline typename TypedImageBase<T>::ImageData&
		TypedImageBase<T>::get_data()
	{
		return this->data;
	}

	template<typename T>
	inline typename TypedImageBase<T>::ImageData const&
		TypedImageBase<T>::get_data() const
	{
		return this->data;
	}

	template<typename T>
	inline T const*
		TypedImageBase<T>::get_data_pointer() const
	{
		if (this->data.empty())
			return nullptr;
		return &this->data[0];
	}

	template<typename T>
	inline T*
		TypedImageBase<T>::get_data_pointer()
	{
		if (this->data.empty())
			return nullptr;
		return &this->data[0];
	}

	template<typename T>
	inline T*
		TypedImageBase<T>::begin()
	{
		return this->data.empty() ? nullptr : &this->data[0];
	}

	template<typename T>
	inline T const*
		TypedImageBase<T>::begin() const
	{
		return this->data.empty() ? nullptr : &this->data[0];
	}

	template<typename T>
	inline T*
		TypedImageBase<T>::end()
	{
		return this->data.empty() ? nullptr : this->begin() + this->data.size();
	}

	template<typename T>
	inline T const*
		TypedImageBase<T>::end() const
	{
		return this->data.empty() ? nullptr : this->begin() + this->data.size();
	}

	template<typename T>
	inline int
		TypedImageBase<T>::get_pixel_amount() const
	{
		return this->w * this->h;
	}

	template<typename T>
	inline int
		TypedImageBase<T>::get_value_amount() const
	{
		return static_cast<int>(this->data.size());
	}

	template<typename T>
	inline std::size_t
		TypedImageBase<T>::get_byte_size() const
	{
		return this->data.size() * sizeof(T);
	}

	template<typename T>
	inline char const*
		TypedImageBase<T>::get_byte_pointer() const
	{
		return reinterpret_cast<char const*>(this->get_data_pointer());
	}

	template<typename T>
	inline char*
		TypedImageBase<T>::get_byte_pointer()
	{
		return reinterpret_cast<char*>(this->get_data_pointer());
	}

}

namespace std
{
	template<class T>
	inline void
		swap(image::TypedImageBase<T>& a, image::TypedImageBase<T>& b)
	{
		a.swap(b);
	}
}