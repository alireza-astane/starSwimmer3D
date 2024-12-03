#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.525618258359418,1.1581320504615686,0.1766514105964077>, 1 }        
    sphere {  m*<0.7666610369518106,1.2716442861251136,3.164792518706963>, 1 }
    sphere {  m*<3.259908226014346,1.271644286125113,-1.0524896897836535>, 1 }
    sphere {  m*<-1.1763464694501078,3.597475617856094,-0.8296494412278503>, 1}
    sphere { m*<-3.9724558201195435,-7.3676634773090885,-2.482238001406837>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7666610369518106,1.2716442861251136,3.164792518706963>, <0.525618258359418,1.1581320504615686,0.1766514105964077>, 0.5 }
    cylinder { m*<3.259908226014346,1.271644286125113,-1.0524896897836535>, <0.525618258359418,1.1581320504615686,0.1766514105964077>, 0.5}
    cylinder { m*<-1.1763464694501078,3.597475617856094,-0.8296494412278503>, <0.525618258359418,1.1581320504615686,0.1766514105964077>, 0.5 }
    cylinder {  m*<-3.9724558201195435,-7.3676634773090885,-2.482238001406837>, <0.525618258359418,1.1581320504615686,0.1766514105964077>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.525618258359418,1.1581320504615686,0.1766514105964077>, 1 }        
    sphere {  m*<0.7666610369518106,1.2716442861251136,3.164792518706963>, 1 }
    sphere {  m*<3.259908226014346,1.271644286125113,-1.0524896897836535>, 1 }
    sphere {  m*<-1.1763464694501078,3.597475617856094,-0.8296494412278503>, 1}
    sphere { m*<-3.9724558201195435,-7.3676634773090885,-2.482238001406837>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7666610369518106,1.2716442861251136,3.164792518706963>, <0.525618258359418,1.1581320504615686,0.1766514105964077>, 0.5 }
    cylinder { m*<3.259908226014346,1.271644286125113,-1.0524896897836535>, <0.525618258359418,1.1581320504615686,0.1766514105964077>, 0.5}
    cylinder { m*<-1.1763464694501078,3.597475617856094,-0.8296494412278503>, <0.525618258359418,1.1581320504615686,0.1766514105964077>, 0.5 }
    cylinder {  m*<-3.9724558201195435,-7.3676634773090885,-2.482238001406837>, <0.525618258359418,1.1581320504615686,0.1766514105964077>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    