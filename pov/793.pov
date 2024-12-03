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
    sphere { m*<8.540932538788025e-19,-5.026306675166222e-18,0.9733374968099471>, 1 }        
    sphere {  m*<5.986647921932824e-20,-3.535745783225069e-18,5.415337496809972>, 1 }
    sphere {  m*<9.428090415820634,-8.084726255400945e-20,-2.359995836523385>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.359995836523385>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.359995836523385>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<5.986647921932824e-20,-3.535745783225069e-18,5.415337496809972>, <8.540932538788025e-19,-5.026306675166222e-18,0.9733374968099471>, 0.5 }
    cylinder { m*<9.428090415820634,-8.084726255400945e-20,-2.359995836523385>, <8.540932538788025e-19,-5.026306675166222e-18,0.9733374968099471>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.359995836523385>, <8.540932538788025e-19,-5.026306675166222e-18,0.9733374968099471>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.359995836523385>, <8.540932538788025e-19,-5.026306675166222e-18,0.9733374968099471>, 0.5}

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
    sphere { m*<8.540932538788025e-19,-5.026306675166222e-18,0.9733374968099471>, 1 }        
    sphere {  m*<5.986647921932824e-20,-3.535745783225069e-18,5.415337496809972>, 1 }
    sphere {  m*<9.428090415820634,-8.084726255400945e-20,-2.359995836523385>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.359995836523385>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.359995836523385>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<5.986647921932824e-20,-3.535745783225069e-18,5.415337496809972>, <8.540932538788025e-19,-5.026306675166222e-18,0.9733374968099471>, 0.5 }
    cylinder { m*<9.428090415820634,-8.084726255400945e-20,-2.359995836523385>, <8.540932538788025e-19,-5.026306675166222e-18,0.9733374968099471>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.359995836523385>, <8.540932538788025e-19,-5.026306675166222e-18,0.9733374968099471>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.359995836523385>, <8.540932538788025e-19,-5.026306675166222e-18,0.9733374968099471>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    