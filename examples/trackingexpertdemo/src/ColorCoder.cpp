#include "ColorCoder.h"


void getcolor(std::uint32_t p, std::uint32_t np, float&r, float&g, float&b) {
    float inc = 6.0 / np;
    float x = p * inc;
    r = 0.0f; g = 0.0f; b = 0.0f;
    if ((0 <= x && x <= 1) || (5 <= x && x <= 6)) r = 1.0f;
    else if (4 <= x && x <= 5) r = x - 4;
    else if (1 <= x && x <= 2) r = 1.0f - (x - 1);
    if (1 <= x && x <= 3) g = 1.0f;
    else if (0 <= x && x <= 1) g = x - 0;
    else if (3 <= x && x <= 4) g = 1.0f - (x - 3);
    if (3 <= x && x <= 5) b = 1.0f;
    else if (2 <= x && x <= 3) b = x - 2;
    else if (5 <= x && x <= 6) b = 1.0f - (x - 5);
}


//static 
void ColorCoder::CPF2Color( std::vector<uint32_t>& desc, std::vector<glm::vec3>& colors)
{

	size_t size = desc.size();

	std::uint32_t max = 0.0;
	std::uint32_t min = 100000000.0;

	for (int i = 0; i < size; i++) {
		if(desc[i] > max) max = desc[i];
		if(desc[i] < min) min = desc[i];
	}

	//cout << "max: " << max << ", min: " << min << endl;

	std::uint32_t range = max - min;

	for (int i = 0; i < size; i++) {
		float value = (float(desc[i]) - min)/float(range);
		float r, g, b;
		getcolor(value  * 255, 255, r, g, b);

		colors.push_back(glm::vec3(r, g, b));
	}

}