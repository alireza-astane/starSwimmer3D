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
    sphere { m*<-1.3978330846500906e-18,-2.4382828411639135e-18,0.6432230430168137>, 1 }        
    sphere {  m*<-2.120289099392581e-18,-5.055323682784053e-18,7.059223043016833>, 1 }
    sphere {  m*<9.428090415820634,-4.1048705009000834e-19,-2.69011029031652>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.69011029031652>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.69011029031652>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.120289099392581e-18,-5.055323682784053e-18,7.059223043016833>, <-1.3978330846500906e-18,-2.4382828411639135e-18,0.6432230430168137>, 0.5 }
    cylinder { m*<9.428090415820634,-4.1048705009000834e-19,-2.69011029031652>, <-1.3978330846500906e-18,-2.4382828411639135e-18,0.6432230430168137>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.69011029031652>, <-1.3978330846500906e-18,-2.4382828411639135e-18,0.6432230430168137>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.69011029031652>, <-1.3978330846500906e-18,-2.4382828411639135e-18,0.6432230430168137>, 0.5}

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
    sphere { m*<-1.3978330846500906e-18,-2.4382828411639135e-18,0.6432230430168137>, 1 }        
    sphere {  m*<-2.120289099392581e-18,-5.055323682784053e-18,7.059223043016833>, 1 }
    sphere {  m*<9.428090415820634,-4.1048705009000834e-19,-2.69011029031652>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.69011029031652>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.69011029031652>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.120289099392581e-18,-5.055323682784053e-18,7.059223043016833>, <-1.3978330846500906e-18,-2.4382828411639135e-18,0.6432230430168137>, 0.5 }
    cylinder { m*<9.428090415820634,-4.1048705009000834e-19,-2.69011029031652>, <-1.3978330846500906e-18,-2.4382828411639135e-18,0.6432230430168137>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.69011029031652>, <-1.3978330846500906e-18,-2.4382828411639135e-18,0.6432230430168137>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.69011029031652>, <-1.3978330846500906e-18,-2.4382828411639135e-18,0.6432230430168137>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    