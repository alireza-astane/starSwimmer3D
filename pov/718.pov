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
    sphere { m*<3.612638757508923e-18,-5.4159910591026686e-18,0.8878759178637105>, 1 }        
    sphere {  m*<1.2253926603832329e-18,-5.22229597941948e-18,5.854875917863733>, 1 }
    sphere {  m*<9.428090415820634,1.1150527793387346e-20,-2.4454574154696216>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.4454574154696216>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.4454574154696216>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2253926603832329e-18,-5.22229597941948e-18,5.854875917863733>, <3.612638757508923e-18,-5.4159910591026686e-18,0.8878759178637105>, 0.5 }
    cylinder { m*<9.428090415820634,1.1150527793387346e-20,-2.4454574154696216>, <3.612638757508923e-18,-5.4159910591026686e-18,0.8878759178637105>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.4454574154696216>, <3.612638757508923e-18,-5.4159910591026686e-18,0.8878759178637105>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.4454574154696216>, <3.612638757508923e-18,-5.4159910591026686e-18,0.8878759178637105>, 0.5}

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
    sphere { m*<3.612638757508923e-18,-5.4159910591026686e-18,0.8878759178637105>, 1 }        
    sphere {  m*<1.2253926603832329e-18,-5.22229597941948e-18,5.854875917863733>, 1 }
    sphere {  m*<9.428090415820634,1.1150527793387346e-20,-2.4454574154696216>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.4454574154696216>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.4454574154696216>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2253926603832329e-18,-5.22229597941948e-18,5.854875917863733>, <3.612638757508923e-18,-5.4159910591026686e-18,0.8878759178637105>, 0.5 }
    cylinder { m*<9.428090415820634,1.1150527793387346e-20,-2.4454574154696216>, <3.612638757508923e-18,-5.4159910591026686e-18,0.8878759178637105>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.4454574154696216>, <3.612638757508923e-18,-5.4159910591026686e-18,0.8878759178637105>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.4454574154696216>, <3.612638757508923e-18,-5.4159910591026686e-18,0.8878759178637105>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    