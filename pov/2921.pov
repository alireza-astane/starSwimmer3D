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
    sphere { m*<0.5664733379346216,1.1050467395106256,0.20080708562621602>, 1 }        
    sphere {  m*<0.8078859668649483,1.2118153359435693,3.1891671341214005>, 1 }
    sphere {  m*<3.3011331559274826,1.2118153359435688,-1.0281150743692145>, 1 }
    sphere {  m*<-1.3462655475005372,3.868498055574018,-0.9301166791029087>, 1}
    sphere { m*<-3.960196795407954,-7.400294520452944,-2.4749891125522057>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8078859668649483,1.2118153359435693,3.1891671341214005>, <0.5664733379346216,1.1050467395106256,0.20080708562621602>, 0.5 }
    cylinder { m*<3.3011331559274826,1.2118153359435688,-1.0281150743692145>, <0.5664733379346216,1.1050467395106256,0.20080708562621602>, 0.5}
    cylinder { m*<-1.3462655475005372,3.868498055574018,-0.9301166791029087>, <0.5664733379346216,1.1050467395106256,0.20080708562621602>, 0.5 }
    cylinder {  m*<-3.960196795407954,-7.400294520452944,-2.4749891125522057>, <0.5664733379346216,1.1050467395106256,0.20080708562621602>, 0.5}

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
    sphere { m*<0.5664733379346216,1.1050467395106256,0.20080708562621602>, 1 }        
    sphere {  m*<0.8078859668649483,1.2118153359435693,3.1891671341214005>, 1 }
    sphere {  m*<3.3011331559274826,1.2118153359435688,-1.0281150743692145>, 1 }
    sphere {  m*<-1.3462655475005372,3.868498055574018,-0.9301166791029087>, 1}
    sphere { m*<-3.960196795407954,-7.400294520452944,-2.4749891125522057>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8078859668649483,1.2118153359435693,3.1891671341214005>, <0.5664733379346216,1.1050467395106256,0.20080708562621602>, 0.5 }
    cylinder { m*<3.3011331559274826,1.2118153359435688,-1.0281150743692145>, <0.5664733379346216,1.1050467395106256,0.20080708562621602>, 0.5}
    cylinder { m*<-1.3462655475005372,3.868498055574018,-0.9301166791029087>, <0.5664733379346216,1.1050467395106256,0.20080708562621602>, 0.5 }
    cylinder {  m*<-3.960196795407954,-7.400294520452944,-2.4749891125522057>, <0.5664733379346216,1.1050467395106256,0.20080708562621602>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    