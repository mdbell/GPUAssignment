
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstdlib>
#include "Image.cuh"

using namespace std;

int readImage(char*, Image&);
int writeImage(char*, Image&);
int readImageHeader(char fname[], int& N, int& M, int& Q, bool& type);

int main(int argc, char* args[]) {
	int scale = 1;
	if (argc < 3) {
		std::cerr << "USAGE: " << args[0] << " INPUT_FILE OUTPUT_FILE [ENLARGE_SIZE]" << std::endl;
		return 1;
	}

	if (argc > 3) {
		scale = atoi(args[3]);
	}

	int N, M, Q;
	bool type;

	readImageHeader(args[1], N, M, Q, type);

	Image img(N, M, Q);
	cout << "Reading: " << args[1] << endl;
	readImage(args[1], img);


	if (scale > 1) {
		cout << "Scaling by a factor of:" << scale << endl;
		img.enlargeImage(scale, img);
	}

	cout << "Negating image" << endl;

	img.negateImage(img);

	cout << "Reflecting image" << endl;

	img.reflectImage(false, img);

	cout << "Writing: " << args[2] << endl;
	writeImage(args[2], img);

	return 0;
}

int readImageHeader(char fname[], int& N, int& M, int& Q, bool& type)
{
	int i, j;
	unsigned char *charImage;
	char header[100], *ptr;
	ifstream ifp;

	ifp.open(fname, ios::in | ios::binary);

	if (!ifp)
	{
		cout << "Can't read image: " << fname << endl;
		exit(1);
	}

	// read header

	type = false; // PGM

	ifp.getline(header, 100, '\n');
	if ((header[0] == 80) && (header[1] == 53))
	{
		type = false;
	}
	else if ((header[0] == 80) && (header[1] == 54))
	{
		type = true;
	}
	else
	{
		cout << "Image " << fname << " is not PGM or PPM" << endl;
		exit(1);
	}

	ifp.getline(header, 100, '\n');
	while (header[0] == '#')
		ifp.getline(header, 100, '\n');

	M = strtol(header, &ptr, 0);
	N = atoi(ptr);

	ifp.getline(header, 100, '\n');

	Q = strtol(header, &ptr, 0);

	ifp.close();

	return(1);
}

int readImage(char fname[], Image& image)
{
	int i, j;
	int N, M, Q;
	unsigned char *charImage;
	int* intImage;
	char header[100], *ptr;
	ifstream ifp;

	ifp.open(fname, ios::in | ios::binary);

	if (!ifp)
	{
		cout << "Can't read image: " << fname << endl;
		exit(1);
	}

	// read header

	ifp.getline(header, 100, '\n');
	if ((header[0] != 80) || (header[1] != 53))
	{
		cout << "Image " << fname << " is not PGM" << endl;
		exit(1);
	}

	ifp.getline(header, 100, '\n');
	while (header[0] == '#')
		ifp.getline(header, 100, '\n');

	M = strtol(header, &ptr, 0);
	N = atoi(ptr);

	ifp.getline(header, 100, '\n');
	Q = strtol(header, &ptr, 0);

	charImage = (unsigned char *) new unsigned char[M*N];
	intImage = new int[M * N];

	ifp.read(reinterpret_cast<char *>(charImage), (M*N) * sizeof(unsigned char));

	if (ifp.fail())
	{
		cout << "Image " << fname << " has wrong size" << endl;
		exit(1);
	}

	ifp.close();

	//
	// Convert the unsigned characters to integers
	//

	int idx;

	for (i = 0; i<N; i++)
		for (j = 0; j<M; j++)
		{
			idx = i*M + j;
			intImage[idx] = (int)charImage[idx];
		}

	image.setPixels(0,0,M * N, intImage);
	delete[] charImage;
	delete[] intImage;

	return (1);

}

int writeImage(char fname[], Image& image)
{
	int i, j;
	int N, M, Q;
	unsigned char *charImage;
	int* intImage;
	ofstream ofp;

	image.getImageInfo(N, M, Q);

	charImage = (unsigned char *) new unsigned char[M*N];
	intImage = new int[M * N];
	image.getPixels(0,0,M * N, intImage); // get rgb pixels
	
	// convert the integer values to unsigned char

	int idx;

	for (i = 0; i<N; i++)
	{
		for (j = 0; j<M; j++)
		{
			idx = i*M + j;
			charImage[idx] = (unsigned char)intImage[idx];
		}
	}

	ofp.open(fname, ios::out | ios::binary);

	if (!ofp)
	{
		cout << "Can't open file: " << fname << endl;
		exit(1);
	}

	ofp << "P5" << endl;
	ofp << M << " " << N << endl;
	ofp << Q << endl;

	ofp.write(reinterpret_cast<char *>(charImage), (M*N) * sizeof(unsigned char));

	if (ofp.fail())
	{
		cout << "Can't write image " << fname << endl;
		exit(0);
	}

	ofp.close();

	delete[] charImage;
	delete[] intImage;
	
	return(1);

}