#ifndef IMAGE_H
#define IMAGE_H

class Image
{
public:
	Image();
	Image(int numRows, int numCols, int grayLevels);
	~Image();
	Image(const Image& oldImage);
	void operator=(const Image&);
	void setImageInfo(int numRows, int numCols, int maxVal);
	void getImageInfo(int &numRows, int &numCols, int &maxVal);
	int getPixelVal(int row, int col);
	void getPixels(int row, int col, int sz, int* out);
	void setPixels(int row, int col, int sz, int* in);
	void setPixelVal(int row, int col, int value);
	void enlargeImage(int value, Image& oldImage);
	void reflectImage(bool flag, Image& oldImage);
	void negateImage(Image& oldImage);
private:
	int N; // number of rows
	int M; // number of columns
	int Q; // number of gray levels
	int *pixelVal;
};

#endif
