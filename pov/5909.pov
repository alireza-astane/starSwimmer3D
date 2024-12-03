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
    sphere { m*<-1.464854371386223,-0.18138501571882948,-1.0977361057789878>, 1 }        
    sphere {  m*<-0.05011237226990617,0.2784371708691043,8.790957114039777>, 1 }
    sphere {  m*<6.825199953552107,0.10349120216781343,-5.455711591177888>, 1 }
    sphere {  m*<-3.138023207332407,2.147890014041536,-1.978060899105319>, 1}
    sphere { m*<-2.8702359862945754,-2.7398019283623616,-1.7885146139427486>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.05011237226990617,0.2784371708691043,8.790957114039777>, <-1.464854371386223,-0.18138501571882948,-1.0977361057789878>, 0.5 }
    cylinder { m*<6.825199953552107,0.10349120216781343,-5.455711591177888>, <-1.464854371386223,-0.18138501571882948,-1.0977361057789878>, 0.5}
    cylinder { m*<-3.138023207332407,2.147890014041536,-1.978060899105319>, <-1.464854371386223,-0.18138501571882948,-1.0977361057789878>, 0.5 }
    cylinder {  m*<-2.8702359862945754,-2.7398019283623616,-1.7885146139427486>, <-1.464854371386223,-0.18138501571882948,-1.0977361057789878>, 0.5}

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
    sphere { m*<-1.464854371386223,-0.18138501571882948,-1.0977361057789878>, 1 }        
    sphere {  m*<-0.05011237226990617,0.2784371708691043,8.790957114039777>, 1 }
    sphere {  m*<6.825199953552107,0.10349120216781343,-5.455711591177888>, 1 }
    sphere {  m*<-3.138023207332407,2.147890014041536,-1.978060899105319>, 1}
    sphere { m*<-2.8702359862945754,-2.7398019283623616,-1.7885146139427486>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.05011237226990617,0.2784371708691043,8.790957114039777>, <-1.464854371386223,-0.18138501571882948,-1.0977361057789878>, 0.5 }
    cylinder { m*<6.825199953552107,0.10349120216781343,-5.455711591177888>, <-1.464854371386223,-0.18138501571882948,-1.0977361057789878>, 0.5}
    cylinder { m*<-3.138023207332407,2.147890014041536,-1.978060899105319>, <-1.464854371386223,-0.18138501571882948,-1.0977361057789878>, 0.5 }
    cylinder {  m*<-2.8702359862945754,-2.7398019283623616,-1.7885146139427486>, <-1.464854371386223,-0.18138501571882948,-1.0977361057789878>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    