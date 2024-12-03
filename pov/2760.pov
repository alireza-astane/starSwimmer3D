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
    sphere { m*<0.6864922621918819,0.942815663477662,0.2717688617364758>, 1 }        
    sphere {  m*<0.9288552017958027,1.0298834933422327,3.260691566412043>, 1 }
    sphere {  m*<3.4221023908583375,1.0298834933422323,-0.9565906420785717>, 1 }
    sphere {  m*<-1.7982161480231307,4.617721020086406,-1.19734054345885>, 1}
    sphere { m*<-3.923284094743942,-7.5020579414254,-2.453162099912677>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9288552017958027,1.0298834933422327,3.260691566412043>, <0.6864922621918819,0.942815663477662,0.2717688617364758>, 0.5 }
    cylinder { m*<3.4221023908583375,1.0298834933422323,-0.9565906420785717>, <0.6864922621918819,0.942815663477662,0.2717688617364758>, 0.5}
    cylinder { m*<-1.7982161480231307,4.617721020086406,-1.19734054345885>, <0.6864922621918819,0.942815663477662,0.2717688617364758>, 0.5 }
    cylinder {  m*<-3.923284094743942,-7.5020579414254,-2.453162099912677>, <0.6864922621918819,0.942815663477662,0.2717688617364758>, 0.5}

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
    sphere { m*<0.6864922621918819,0.942815663477662,0.2717688617364758>, 1 }        
    sphere {  m*<0.9288552017958027,1.0298834933422327,3.260691566412043>, 1 }
    sphere {  m*<3.4221023908583375,1.0298834933422323,-0.9565906420785717>, 1 }
    sphere {  m*<-1.7982161480231307,4.617721020086406,-1.19734054345885>, 1}
    sphere { m*<-3.923284094743942,-7.5020579414254,-2.453162099912677>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9288552017958027,1.0298834933422327,3.260691566412043>, <0.6864922621918819,0.942815663477662,0.2717688617364758>, 0.5 }
    cylinder { m*<3.4221023908583375,1.0298834933422323,-0.9565906420785717>, <0.6864922621918819,0.942815663477662,0.2717688617364758>, 0.5}
    cylinder { m*<-1.7982161480231307,4.617721020086406,-1.19734054345885>, <0.6864922621918819,0.942815663477662,0.2717688617364758>, 0.5 }
    cylinder {  m*<-3.923284094743942,-7.5020579414254,-2.453162099912677>, <0.6864922621918819,0.942815663477662,0.2717688617364758>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    