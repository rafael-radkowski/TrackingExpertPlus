#version 410 core     
#define MAX_LIGHTS 4

uniform mat4 projectionMatrix;                                    
uniform mat4 viewMatrix;                                           
uniform mat4 modelMatrix;  
                                                                   
in vec3 pass_Normal;     
in vec3 pass_Position;   
in vec4 pass_Color;   

// The material parameters
uniform struct LightSource {
    vec3 position;
	vec3 direction;
	vec3 color;
	float intensity;
	float cutoff_in; // inner
	float cutoff_out; // outer
	float k1;    //attenuation
	float k2;    //attenuation
	bool used;
	int  type;  //0:point, 1:spot, 2:directional
} light[MAX_LIGHTS];


// The material parameters
uniform struct Material {
    vec3  diffColor;
 	float diffInt;
 	vec3  ambColor;
 	float ambInt;
 	vec3  specColor;
 	float specInt;
 	float shininess;
} mat[1];


out vec4 color;       

/*
Per-fragment light. 
Note that all vectors need to be in camera/eye-space. 
@param L - vector between light and fragment position.
@param E - eye point
@param s - the light source
@param m - the material 
*/
vec4 useLight(vec3 L, vec3 E, vec3 N, LightSource s, Material m)
{
	vec4 color = vec4(0.0,0.0,0.0,0.0);

	// diffuse light
	vec3 Idiff =  m.diffInt * m.diffColor *  max(dot(L, N), 0.0); 
	Idiff = clamp(Idiff, 0.0, 1.0); 

	// ambient light
	vec4 Iamb = vec4( m.ambColor, 1.0) * m.ambInt;

	// specular light
	vec3 R = reflect(L, N);
	vec3 Ispec =  m.specInt * m.specColor *  pow(max(dot(R, E), 0.0), m.shininess); 

	// calculate color                                     
	color = max(vec4( ( Idiff + Ispec) * s.color, 1.0), Iamb)  * s.intensity;  

	// attenuation 
	float dist = length(L);
	float Iatt = 1 / (1 + s.k1 * dist + s.k2*s.k2 * dist);

	if(s.type == 0)//pointlight
	{
		color = color * Iatt;
	}
	else if(s.type == 1)// spotlight
	{
		vec4 K = viewMatrix * vec4(normalize(s.direction - s.position), 0.0);
		float a = dot(-L,K.xyz); // angle between light dir and fragment position. 

		float c = smoothstep(1.0-s.cutoff_out, 1.0-s.cutoff_in, a); 
		Iatt = Iatt * c; // multiple with attenuation to maintain the distance effect. 

		color = color * Iatt; // attenutation
	}


	return color;
}

                       
void main(void)                                                  
{    
	// eye position 
	vec3 E = normalize( vec3(viewMatrix[3][0], viewMatrix[3][1], viewMatrix[3][2]) );
     

	// calc light. 
	vec4 mixed = vec4(0.0,0.0,0.0,1.0);
	for (int i=0; i<MAX_LIGHTS; i++){

		if(light[i].used == false) continue;

		// light position if camera frame
		vec4 L_c = viewMatrix * vec4(light[i].position, 1.0);
	
		// light to fragment 
		vec3 L = normalize( L_c.xyz - pass_Position );
		if(light[i].type == 2) L = light[i].direction;// direct light

		// checks whether the light was set.
		// Multiple lights blend adative
		mixed += useLight( L,  E,  pass_Normal, light[i], mat[0]);
		
	}
	                                             
	color = mixed;                         
}                                                      