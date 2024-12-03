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
    sphere { m*<-4.46072955793495e-18,7.82094356852084e-19,0.14304662381632252>, 1 }        
    sphere {  m*<-4.7276060105646806e-18,-2.3951377284968794e-18,9.36604662381632>, 1 }
    sphere {  m*<9.428090415820634,-2.7060405131339042e-18,-3.1902867095170118>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.1902867095170118>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.1902867095170118>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.7276060105646806e-18,-2.3951377284968794e-18,9.36604662381632>, <-4.46072955793495e-18,7.82094356852084e-19,0.14304662381632252>, 0.5 }
    cylinder { m*<9.428090415820634,-2.7060405131339042e-18,-3.1902867095170118>, <-4.46072955793495e-18,7.82094356852084e-19,0.14304662381632252>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.1902867095170118>, <-4.46072955793495e-18,7.82094356852084e-19,0.14304662381632252>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.1902867095170118>, <-4.46072955793495e-18,7.82094356852084e-19,0.14304662381632252>, 0.5}

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
    sphere { m*<-4.46072955793495e-18,7.82094356852084e-19,0.14304662381632252>, 1 }        
    sphere {  m*<-4.7276060105646806e-18,-2.3951377284968794e-18,9.36604662381632>, 1 }
    sphere {  m*<9.428090415820634,-2.7060405131339042e-18,-3.1902867095170118>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.1902867095170118>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.1902867095170118>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.7276060105646806e-18,-2.3951377284968794e-18,9.36604662381632>, <-4.46072955793495e-18,7.82094356852084e-19,0.14304662381632252>, 0.5 }
    cylinder { m*<9.428090415820634,-2.7060405131339042e-18,-3.1902867095170118>, <-4.46072955793495e-18,7.82094356852084e-19,0.14304662381632252>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.1902867095170118>, <-4.46072955793495e-18,7.82094356852084e-19,0.14304662381632252>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.1902867095170118>, <-4.46072955793495e-18,7.82094356852084e-19,0.14304662381632252>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    