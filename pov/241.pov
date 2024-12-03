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
    sphere { m*<-5.271970416193597e-18,5.721511088919384e-20,0.3095410979131732>, 1 }        
    sphere {  m*<-1.0006816957151751e-17,-2.890339423635088e-18,8.615541097913185>, 1 }
    sphere {  m*<9.428090415820634,-2.0553844770755662e-18,-3.023792235420159>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.023792235420159>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.023792235420159>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.0006816957151751e-17,-2.890339423635088e-18,8.615541097913185>, <-5.271970416193597e-18,5.721511088919384e-20,0.3095410979131732>, 0.5 }
    cylinder { m*<9.428090415820634,-2.0553844770755662e-18,-3.023792235420159>, <-5.271970416193597e-18,5.721511088919384e-20,0.3095410979131732>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.023792235420159>, <-5.271970416193597e-18,5.721511088919384e-20,0.3095410979131732>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.023792235420159>, <-5.271970416193597e-18,5.721511088919384e-20,0.3095410979131732>, 0.5}

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
    sphere { m*<-5.271970416193597e-18,5.721511088919384e-20,0.3095410979131732>, 1 }        
    sphere {  m*<-1.0006816957151751e-17,-2.890339423635088e-18,8.615541097913185>, 1 }
    sphere {  m*<9.428090415820634,-2.0553844770755662e-18,-3.023792235420159>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.023792235420159>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.023792235420159>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.0006816957151751e-17,-2.890339423635088e-18,8.615541097913185>, <-5.271970416193597e-18,5.721511088919384e-20,0.3095410979131732>, 0.5 }
    cylinder { m*<9.428090415820634,-2.0553844770755662e-18,-3.023792235420159>, <-5.271970416193597e-18,5.721511088919384e-20,0.3095410979131732>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.023792235420159>, <-5.271970416193597e-18,5.721511088919384e-20,0.3095410979131732>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.023792235420159>, <-5.271970416193597e-18,5.721511088919384e-20,0.3095410979131732>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    