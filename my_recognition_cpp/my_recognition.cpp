#include <jetson-inference/imageNet.h>
#include <jetson-utils/loadImage.h>

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("my-recognition: expected image filename as argument\n");
		printf("example usage : ./my_recognition my_image.jpg\n");
		return 0;
	}

	const char* imgFilename = argv[1];
	
	// Loading the Image from Disk
	float* imgCPU = NULL;	// CPU pointer to floating-point RGBA image data
	float* imgCUDA = NULL;	// GPU pointer to floating-point RGBA image data
	int imgWidth = 0;		// width of the image (in pixels)
	int imgHeight = 0;		// height of the image (in pixels)

	// load the image from disk as float4 RGBA (32 bits per channel, 128 bits per pixel)
	if (!loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight))
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}
	
	// Loading the Image Recognition Network
	imageNet* net = imageNet::Create(imageNet::GOOGLENET);
	if (!net)
	{
		printf("failed to load image recognition network\n");
		return 0;
	}

	// Classifying the Image
	float confidence = 0.0;
	const int classIndex = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);

	// Intepreting the Results
	if (classIndex >= 0)
	{
		const char* classDescription = net->GetClassDesc(classIndex);

		printf("image is recognized as '%s' (class #%i) with %f%% confidence\n", classDescription, classIndex, confidence * 100.0f);
	}
	else
	{
		printf("failed to classify image\n");
	}

	delete net;

	return 0;
}

