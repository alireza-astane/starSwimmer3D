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
    sphere { m*<5.257451498814473e-19,-4.9691823707270306e-18,0.969957706394244>, 1 }        
    sphere {  m*<9.283557884535761e-20,-3.59153248489919e-18,5.4329577063942684>, 1 }
    sphere {  m*<9.428090415820634,-2.6152385429842062e-20,-2.363375626939088>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.363375626939088>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.363375626939088>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<9.283557884535761e-20,-3.59153248489919e-18,5.4329577063942684>, <5.257451498814473e-19,-4.9691823707270306e-18,0.969957706394244>, 0.5 }
    cylinder { m*<9.428090415820634,-2.6152385429842062e-20,-2.363375626939088>, <5.257451498814473e-19,-4.9691823707270306e-18,0.969957706394244>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.363375626939088>, <5.257451498814473e-19,-4.9691823707270306e-18,0.969957706394244>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.363375626939088>, <5.257451498814473e-19,-4.9691823707270306e-18,0.969957706394244>, 0.5}

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
    sphere { m*<5.257451498814473e-19,-4.9691823707270306e-18,0.969957706394244>, 1 }        
    sphere {  m*<9.283557884535761e-20,-3.59153248489919e-18,5.4329577063942684>, 1 }
    sphere {  m*<9.428090415820634,-2.6152385429842062e-20,-2.363375626939088>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.363375626939088>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.363375626939088>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<9.283557884535761e-20,-3.59153248489919e-18,5.4329577063942684>, <5.257451498814473e-19,-4.9691823707270306e-18,0.969957706394244>, 0.5 }
    cylinder { m*<9.428090415820634,-2.6152385429842062e-20,-2.363375626939088>, <5.257451498814473e-19,-4.9691823707270306e-18,0.969957706394244>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.363375626939088>, <5.257451498814473e-19,-4.9691823707270306e-18,0.969957706394244>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.363375626939088>, <5.257451498814473e-19,-4.9691823707270306e-18,0.969957706394244>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    