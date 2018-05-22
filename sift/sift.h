#pragma once
#include <vector>

#include "image.h"
#include "vector.h"


class Sift
{
public:

	//SIFT�������
	struct Config
	{
		Config();


		/*
			ÿ��octave���ĳ߶�����Ĭ��Ϊ3ʱ��
			��6���˹�ռ䣬5��DoG�ռ�
		*/
		int num_samples_per_octave;

		/*
			��Сoctave����
			Ĭ��Ϊ0��������ͼ��ߴ���Ϊ�����ߴ�
			������Ὣͼ��downsample������Ϊ2
			����Ϊ-1ʱ���Ŵ�ͼƬ������Ϊ2
		*/
		int min_octave;

		/*
			���octave����Ĭ��Ϊ4����ԭͼ�Ĵ���С
		*/
		int max_octave;

		/*
			�ؼ����ֵʱDoG��ֵ
			Ĭ��Ϊ0.02/samples
		*/
		float contrast_threshold;

		/*
			������Ե��Ӧ����ֵ��Ϊ������r��Ĭ��Ϊ10
		*/
		float edge_ratio_threshold;

		/*
			�����µ�octaveʱҪ���ģ��ϵ��
			Ĭ�� sigma = 1.6
		*/
		float base_blur_sigma;

		/*
			����ͼ������ģ��ϵ����Ĭ��0.5
		*/
		float inherent_blur_sigma;

		/*
		   �Ƿ������̨���״̬��Ϣ
		*/
		bool verbose_output;

		/*
			�Ƿ������̨���������Ϣ
		*/
		bool debug_output;
	};

	/*
		SIFT�ؼ����representation
	*/
	struct Keypoint
	{
		//keypoint��octave����
		int octave;
		//sample����
		float sample;
		//keypoint x����
		float x;
		//keypoint y����
		float y;
	};

	/*
		SIFT�����ӵ�representation
	*/
	struct Descriptor
	{
		//�ؼ���������x����
		float x;
		//�ؼ���������y����
		float y;
		//�ؼ���߶�(sigmaֵ)
		float scale;
		//�ؼ��㷽�� {0��2PI]
		float orientation;
		//���������ݣ��޷��� [0.0, 1.0]
		math::Vector<float, 128> data;
	};

public:
	typedef std::vector<Keypoint> Keypoints;
	typedef std::vector<Descriptor> Descriptors;

public:
	explicit Sift(Config const& conf);

	//��������ͼ��
	void set_image(image::ByteImage::ConstPtr img);
	//��������ͼ��
	void set_float_image(image::FloatImage::ConstPtr img);

	//��ʼSIFT�ؼ��������������ȡ
	void process();

	//���عؼ��㼯��
	Keypoints const& get_keypoints() const;
	//���������Ӽ���
	Descriptors const& get_descriptors() const;


	static void load_lowe_descriptor(std::string const& filename,
		Descriptor* result);

protected:
	/*
		SIFT octave��representation
	*/
	struct Octave
	{
		typedef std::vector<image::FloatImage::Ptr> ImageVector;
		ImageVector img;	///< S+3 images per octave				ÿ��octave�ĸ�˹�ռ�
		ImageVector dog;	///< S+2 difference of gaussian images  ��˹��ֿռ�
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
	//ԭʼ����ͼƬ
	image::FloatImage::ConstPtr orig;
	//ͼ�������
	Octaves octaves;
	//��⵽�Ĺؼ���
	Keypoints keypoints;
	//���յ�SIFT������
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

