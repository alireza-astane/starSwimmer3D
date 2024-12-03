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
    sphere { m*<-1.9004740863764358e-18,-5.4118271841884764e-18,1.0990059381154709>, 1 }        
    sphere {  m*<-3.930075302227932e-18,-4.750669924363166e-18,4.743005938115509>, 1 }
    sphere {  m*<9.428090415820634,2.1207196331592017e-19,-2.234327395217862>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.234327395217862>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.234327395217862>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.930075302227932e-18,-4.750669924363166e-18,4.743005938115509>, <-1.9004740863764358e-18,-5.4118271841884764e-18,1.0990059381154709>, 0.5 }
    cylinder { m*<9.428090415820634,2.1207196331592017e-19,-2.234327395217862>, <-1.9004740863764358e-18,-5.4118271841884764e-18,1.0990059381154709>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.234327395217862>, <-1.9004740863764358e-18,-5.4118271841884764e-18,1.0990059381154709>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.234327395217862>, <-1.9004740863764358e-18,-5.4118271841884764e-18,1.0990059381154709>, 0.5}

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
    sphere { m*<-1.9004740863764358e-18,-5.4118271841884764e-18,1.0990059381154709>, 1 }        
    sphere {  m*<-3.930075302227932e-18,-4.750669924363166e-18,4.743005938115509>, 1 }
    sphere {  m*<9.428090415820634,2.1207196331592017e-19,-2.234327395217862>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.234327395217862>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.234327395217862>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.930075302227932e-18,-4.750669924363166e-18,4.743005938115509>, <-1.9004740863764358e-18,-5.4118271841884764e-18,1.0990059381154709>, 0.5 }
    cylinder { m*<9.428090415820634,2.1207196331592017e-19,-2.234327395217862>, <-1.9004740863764358e-18,-5.4118271841884764e-18,1.0990059381154709>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.234327395217862>, <-1.9004740863764358e-18,-5.4118271841884764e-18,1.0990059381154709>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.234327395217862>, <-1.9004740863764358e-18,-5.4118271841884764e-18,1.0990059381154709>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    