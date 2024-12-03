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
    sphere { m*<-5.726024800594949e-19,-5.369175060942738e-18,1.1490091061367662>, 1 }        
    sphere {  m*<-4.811053849195223e-18,-5.1023230496661885e-18,4.464009106136808>, 1 }
    sphere {  m*<9.428090415820634,7.370695152960794e-20,-2.1843242271965666>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.1843242271965666>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.1843242271965666>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.811053849195223e-18,-5.1023230496661885e-18,4.464009106136808>, <-5.726024800594949e-19,-5.369175060942738e-18,1.1490091061367662>, 0.5 }
    cylinder { m*<9.428090415820634,7.370695152960794e-20,-2.1843242271965666>, <-5.726024800594949e-19,-5.369175060942738e-18,1.1490091061367662>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.1843242271965666>, <-5.726024800594949e-19,-5.369175060942738e-18,1.1490091061367662>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.1843242271965666>, <-5.726024800594949e-19,-5.369175060942738e-18,1.1490091061367662>, 0.5}

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
    sphere { m*<-5.726024800594949e-19,-5.369175060942738e-18,1.1490091061367662>, 1 }        
    sphere {  m*<-4.811053849195223e-18,-5.1023230496661885e-18,4.464009106136808>, 1 }
    sphere {  m*<9.428090415820634,7.370695152960794e-20,-2.1843242271965666>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.1843242271965666>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.1843242271965666>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.811053849195223e-18,-5.1023230496661885e-18,4.464009106136808>, <-5.726024800594949e-19,-5.369175060942738e-18,1.1490091061367662>, 0.5 }
    cylinder { m*<9.428090415820634,7.370695152960794e-20,-2.1843242271965666>, <-5.726024800594949e-19,-5.369175060942738e-18,1.1490091061367662>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.1843242271965666>, <-5.726024800594949e-19,-5.369175060942738e-18,1.1490091061367662>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.1843242271965666>, <-5.726024800594949e-19,-5.369175060942738e-18,1.1490091061367662>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    