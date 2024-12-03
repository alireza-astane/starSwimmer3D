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
    sphere { m*<-0.8472804820095309,-1.224356164579991,-0.6439828376304324>, 1 }        
    sphere {  m*<0.5772093915854086,-0.34316350233544834,9.214803919503733>, 1 }
    sphere {  m*<7.932560829585383,-0.4320837783298044,-5.364689370541596>, 1 }
    sphere {  m*<-6.788509606201094,5.784500207222846,-3.684292721187315>, 1}
    sphere { m*<-2.066709004666995,-3.898806046291047,-1.2428675805533007>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5772093915854086,-0.34316350233544834,9.214803919503733>, <-0.8472804820095309,-1.224356164579991,-0.6439828376304324>, 0.5 }
    cylinder { m*<7.932560829585383,-0.4320837783298044,-5.364689370541596>, <-0.8472804820095309,-1.224356164579991,-0.6439828376304324>, 0.5}
    cylinder { m*<-6.788509606201094,5.784500207222846,-3.684292721187315>, <-0.8472804820095309,-1.224356164579991,-0.6439828376304324>, 0.5 }
    cylinder {  m*<-2.066709004666995,-3.898806046291047,-1.2428675805533007>, <-0.8472804820095309,-1.224356164579991,-0.6439828376304324>, 0.5}

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
    sphere { m*<-0.8472804820095309,-1.224356164579991,-0.6439828376304324>, 1 }        
    sphere {  m*<0.5772093915854086,-0.34316350233544834,9.214803919503733>, 1 }
    sphere {  m*<7.932560829585383,-0.4320837783298044,-5.364689370541596>, 1 }
    sphere {  m*<-6.788509606201094,5.784500207222846,-3.684292721187315>, 1}
    sphere { m*<-2.066709004666995,-3.898806046291047,-1.2428675805533007>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5772093915854086,-0.34316350233544834,9.214803919503733>, <-0.8472804820095309,-1.224356164579991,-0.6439828376304324>, 0.5 }
    cylinder { m*<7.932560829585383,-0.4320837783298044,-5.364689370541596>, <-0.8472804820095309,-1.224356164579991,-0.6439828376304324>, 0.5}
    cylinder { m*<-6.788509606201094,5.784500207222846,-3.684292721187315>, <-0.8472804820095309,-1.224356164579991,-0.6439828376304324>, 0.5 }
    cylinder {  m*<-2.066709004666995,-3.898806046291047,-1.2428675805533007>, <-0.8472804820095309,-1.224356164579991,-0.6439828376304324>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    