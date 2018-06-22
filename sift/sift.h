#pragma once
#include <vector>

#include "image.h"
#include "vector.h"


class Sift
{
public:

	//SIFT检测配置
	struct Config
	{
		Config();


		/*
			每组octave检测的尺度数，默认为3时，
			有6层高斯空间，5层DoG空间
		*/
		int num_samples_per_octave;

		/*
			最小octave索引
			默认为0，把输入图像尺寸作为基础尺寸
			大于零会将图像downsample，参数为2
			设置为-1时，放大图片，参数为2
		*/
		int min_octave;

		/*
			最大octave数，默认为4，对原图四次缩小
		*/
		int max_octave;

		/*
			关键点插值时DoG阈值
			默认为0.02/samples
		*/
		float contrast_threshold;

		/*
			消除边缘响应的阈值，为主曲率r，默认为10
		*/
		float edge_ratio_threshold;

		/*
			建立新的octave时要求的模糊系数
			默认 sigma = 1.6
		*/
		float base_blur_sigma;

		/*
			输入图像内在模糊系数，默认0.5
		*/
		float inherent_blur_sigma;

		/*
		   是否向控制台输出状态信息
		*/
		bool verbose_output;

		/*
			是否向控制台输出更多信息
		*/
		bool debug_output;
	};

	/*
		SIFT关键点的representation
	*/
	struct Keypoint
	{
		//keypoint的octave索引
		int octave;
		//sample索引
		float sample;
		//keypoint x坐标
		float x;
		//keypoint y坐标
		float y;
	};

	/*
		SIFT描述子的representation
	*/
	struct Descriptor
	{
		//关键点亚像素x坐标
		float x;
		//关键点亚像素y坐标
		float y;
		//关键点尺度(sigma值)
		float scale;
		//关键点方向 {0，2PI]
		float orientation;
		//描述子数据，无符号 [0.0, 1.0]
		math::Vector<float, 128> data;
	};

public:
	typedef std::vector<Keypoint> Keypoints;
	typedef std::vector<Descriptor> Descriptors;

public:
	explicit Sift(Config const& conf);

	//设置输入图像
	void set_image(image::ByteImage::ConstPtr img);
	//设置输入图像
	void set_float_image(image::FloatImage::ConstPtr img);

	//开始SIFT关键点检测和描述子提取
	void process();

	//返回关键点集合
	Keypoints const& get_keypoints() const;
	//返回描述子集合
	Descriptors const& get_descriptors() const;


	static void load_lowe_descriptor(std::string const& filename,
		Descriptor* result);

protected:
	/*
		SIFT octave的representation
	*/
	struct Octave
	{
		typedef std::vector<image::FloatImage::Ptr> ImageVector;
		ImageVector img;	///< S+3 images per octave				每个octave的高斯空间
		ImageVector dog;	///< S+2 difference of gaussian images  高斯差分空间
		ImageVector grad;	///< S+3 gradient images
		ImageVector ori;	///< S+3 orientation images
	};

protected:
	typedef std::vector<Octave> Octaves;

protected:
	void create_octaves();
	void add_octave(image::FloatImage::ConstPtr image,
		float has_sigma, float target_sigma);
	void extrema_detection();
	std::size_t extrema_detection(image::FloatImage::ConstPtr s[3],
		int oi, int si);
	void keypoint_localization();

	void descriptor_generation();
	void generate_grad_ori_images(Octave* octave);
	void orientation_assignment(Keypoint const& kp,
		Octave const* octave, std::vector<float>& orientations);
	bool descriptor_assignment(Keypoint const& kp, Descriptor& desc,
		Octave const* octave);

	float keypoint_relative_scale(Keypoint const& kp);
	float keypoint_absolute_scale(Keypoint const& kp);

private:
	Config config;
	//原始输入图片
	image::FloatImage::ConstPtr orig;
	//图像金字塔
	Octaves octaves;
	//检测到的关键点
	Keypoints keypoints;
	//最终的SIFT描述子
	Descriptors descriptors;
};

inline
Sift::Config::Config() :
	num_samples_per_octave(3),
	min_octave(0),
	max_octave(4),
	contrast_threshold(-1.0f),
	edge_ratio_threshold(10.0f),
	base_blur_sigma(1.6f),
	inherent_blur_sigma(0.5f),
	verbose_output(true),
	debug_output(true)
{
}

inline Sift::Keypoints const&
Sift::get_keypoints() const
{
	return this->keypoints;
}

inline Sift::Descriptors const&
Sift::get_descriptors() const
{
	return this->descriptors;
}

