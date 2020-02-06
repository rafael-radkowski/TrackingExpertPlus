#include "DemoScene.h"


std::vector<std::string> g_files = {
	"../examples/gear_demo/models/N0-000_pc_gfx.obj"
};


DemoScene::DemoScene()
{

}

DemoScene::~DemoScene()
{

}


/*
Create the demo scene
*/
void DemoScene::create(void)
{
	for(auto m : g_files){
		_3d_models.push_back(cs557::OBJModel());
		_3d_models.back().create(m);
	}
	_3d_models[0].setModelMatrix(glm::translate(glm::vec3(0.0,0.0,0.4)));
}



void DemoScene::draw(glm::mat4 projectionMatrix, glm::mat4 viewMatrix)
{
	for (auto i : _3d_models) {
		i.draw(projectionMatrix, viewMatrix);
	}
}