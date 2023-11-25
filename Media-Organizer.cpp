// Media-Organizer.cpp : Defines the entry point for the application.
//

#include "Media-Organizer.h"
#include <iostream>
#include "dlib/dnn.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/string.h>
#include <dlib/clustering.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <fstream>

using namespace std;
using namespace dlib;

//ResNet Code

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

int face_uptree_parents[2500000];
int array_fill_count = 0;
std::vector<rectangle> identified_face_rectangles;
std::vector<matrix<float, 0, 1>> face_descriptors_history;

void recognize_faces(string path, frontal_face_detector& face_detector, shape_predictor& shape_predictor, anet_type& net) {
	matrix<rgb_pixel> og_image;
	load_image(og_image, path);
	std::vector<matrix<rgb_pixel>> faces;

	for (auto face : face_detector(og_image)) {

		auto shape = shape_predictor(og_image, face);
		face_uptree_parents[array_fill_count] = -1;
		identified_face_rectangles.push_back(face);


		std::cout << face_uptree_parents[array_fill_count] << std::endl;
		std::cout << identified_face_rectangles[array_fill_count] << std::endl;

		matrix<rgb_pixel> face_chip;
		extract_image_chip(og_image, get_face_chip_details(shape, 150, 0.25), face_chip);
		faces.push_back(move(face_chip));

		array_fill_count++;
	}

	std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
	face_descriptors_history.insert(face_descriptors_history.end(),
		face_descriptors.begin(),
		face_descriptors.end());
	cout << "des" << face_descriptors.size() << endl;
	int k = 0;
	for (size_t i = 0; i < face_descriptors.size(); i++)
	{
		for (int k = 0; k <= array_fill_count; k++) {
			
			if (face_uptree_parents[k] == -1) {
				if (length(face_descriptors_history[k] - face_descriptors[i]) < 0.6) {
					face_uptree_parents[array_fill_count + i] = k;
					cout << "match" << endl;

				}
				else {
					cout << "no match" << endl;
				}
			}
		}
	}

}


int main()
{

	frontal_face_detector face_detector = get_frontal_face_detector();
	shape_predictor shape_predictor;
	deserialize("shape_predictor_5_face_landmarks.dat") >> shape_predictor;
	anet_type net;
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	recognize_faces("3.jpg", face_detector, shape_predictor, net);
	recognize_faces("tanay.jpg", face_detector, shape_predictor, net);
	recognize_faces("deep_nitesh.jpg", face_detector, shape_predictor, net);

	for (int i = 0; i < 20; i++) {

		cout << face_uptree_parents[i] << endl;

	}
	cout << array_fill_count << endl;



}
