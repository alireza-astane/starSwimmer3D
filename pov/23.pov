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
    sphere { m*<3.393091087867544e-19,1.5194308582916532e-18,0.031072359914691768>, 1 }        
    sphere {  m*<-9.498423916901386e-19,5.013104125498241e-19,9.86307235991469>, 1 }
    sphere {  m*<9.428090415820634,3.958390797106294e-20,-3.3022609734186426>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.3022609734186426>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.3022609734186426>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-9.498423916901386e-19,5.013104125498241e-19,9.86307235991469>, <3.393091087867544e-19,1.5194308582916532e-18,0.031072359914691768>, 0.5 }
    cylinder { m*<9.428090415820634,3.958390797106294e-20,-3.3022609734186426>, <3.393091087867544e-19,1.5194308582916532e-18,0.031072359914691768>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.3022609734186426>, <3.393091087867544e-19,1.5194308582916532e-18,0.031072359914691768>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.3022609734186426>, <3.393091087867544e-19,1.5194308582916532e-18,0.031072359914691768>, 0.5}

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
    sphere { m*<3.393091087867544e-19,1.5194308582916532e-18,0.031072359914691768>, 1 }        
    sphere {  m*<-9.498423916901386e-19,5.013104125498241e-19,9.86307235991469>, 1 }
    sphere {  m*<9.428090415820634,3.958390797106294e-20,-3.3022609734186426>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.3022609734186426>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.3022609734186426>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-9.498423916901386e-19,5.013104125498241e-19,9.86307235991469>, <3.393091087867544e-19,1.5194308582916532e-18,0.031072359914691768>, 0.5 }
    cylinder { m*<9.428090415820634,3.958390797106294e-20,-3.3022609734186426>, <3.393091087867544e-19,1.5194308582916532e-18,0.031072359914691768>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.3022609734186426>, <3.393091087867544e-19,1.5194308582916532e-18,0.031072359914691768>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.3022609734186426>, <3.393091087867544e-19,1.5194308582916532e-18,0.031072359914691768>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    