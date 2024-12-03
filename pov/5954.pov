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
    sphere { m*<-1.527943712078986,-0.18345003075658348,-1.0619085822015841>, 1 }        
    sphere {  m*<-0.08730912701243695,0.2776413692766326,8.82298454182253>, 1 }
    sphere {  m*<7.029040946235565,0.10949294436028109,-5.588801611807498>, 1 }
    sphere {  m*<-3.203172891130678,2.1458686921830346,-1.9381898530979556>, 1}
    sphere { m*<-2.9353856700928467,-2.741823250220863,-1.7486435679353851>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.08730912701243695,0.2776413692766326,8.82298454182253>, <-1.527943712078986,-0.18345003075658348,-1.0619085822015841>, 0.5 }
    cylinder { m*<7.029040946235565,0.10949294436028109,-5.588801611807498>, <-1.527943712078986,-0.18345003075658348,-1.0619085822015841>, 0.5}
    cylinder { m*<-3.203172891130678,2.1458686921830346,-1.9381898530979556>, <-1.527943712078986,-0.18345003075658348,-1.0619085822015841>, 0.5 }
    cylinder {  m*<-2.9353856700928467,-2.741823250220863,-1.7486435679353851>, <-1.527943712078986,-0.18345003075658348,-1.0619085822015841>, 0.5}

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
    sphere { m*<-1.527943712078986,-0.18345003075658348,-1.0619085822015841>, 1 }        
    sphere {  m*<-0.08730912701243695,0.2776413692766326,8.82298454182253>, 1 }
    sphere {  m*<7.029040946235565,0.10949294436028109,-5.588801611807498>, 1 }
    sphere {  m*<-3.203172891130678,2.1458686921830346,-1.9381898530979556>, 1}
    sphere { m*<-2.9353856700928467,-2.741823250220863,-1.7486435679353851>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.08730912701243695,0.2776413692766326,8.82298454182253>, <-1.527943712078986,-0.18345003075658348,-1.0619085822015841>, 0.5 }
    cylinder { m*<7.029040946235565,0.10949294436028109,-5.588801611807498>, <-1.527943712078986,-0.18345003075658348,-1.0619085822015841>, 0.5}
    cylinder { m*<-3.203172891130678,2.1458686921830346,-1.9381898530979556>, <-1.527943712078986,-0.18345003075658348,-1.0619085822015841>, 0.5 }
    cylinder {  m*<-2.9353856700928467,-2.741823250220863,-1.7486435679353851>, <-1.527943712078986,-0.18345003075658348,-1.0619085822015841>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    