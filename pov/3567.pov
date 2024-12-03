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
    sphere { m*<0.10303700744140204,0.4026550926514324,-0.068351465260164>, 1 }        
    sphere {  m*<0.34377211218309356,0.5313651708317578,2.9192033058603855>, 1 }
    sphere {  m*<2.83774540144766,0.5046890680378067,-1.2975609907113497>, 1 }
    sphere {  m*<-1.518578352451489,2.7311290370700334,-1.0422972306761351>, 1}
    sphere { m*<-2.6154498953795495,-4.736254647303032,-1.643425621608975>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.34377211218309356,0.5313651708317578,2.9192033058603855>, <0.10303700744140204,0.4026550926514324,-0.068351465260164>, 0.5 }
    cylinder { m*<2.83774540144766,0.5046890680378067,-1.2975609907113497>, <0.10303700744140204,0.4026550926514324,-0.068351465260164>, 0.5}
    cylinder { m*<-1.518578352451489,2.7311290370700334,-1.0422972306761351>, <0.10303700744140204,0.4026550926514324,-0.068351465260164>, 0.5 }
    cylinder {  m*<-2.6154498953795495,-4.736254647303032,-1.643425621608975>, <0.10303700744140204,0.4026550926514324,-0.068351465260164>, 0.5}

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
    sphere { m*<0.10303700744140204,0.4026550926514324,-0.068351465260164>, 1 }        
    sphere {  m*<0.34377211218309356,0.5313651708317578,2.9192033058603855>, 1 }
    sphere {  m*<2.83774540144766,0.5046890680378067,-1.2975609907113497>, 1 }
    sphere {  m*<-1.518578352451489,2.7311290370700334,-1.0422972306761351>, 1}
    sphere { m*<-2.6154498953795495,-4.736254647303032,-1.643425621608975>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.34377211218309356,0.5313651708317578,2.9192033058603855>, <0.10303700744140204,0.4026550926514324,-0.068351465260164>, 0.5 }
    cylinder { m*<2.83774540144766,0.5046890680378067,-1.2975609907113497>, <0.10303700744140204,0.4026550926514324,-0.068351465260164>, 0.5}
    cylinder { m*<-1.518578352451489,2.7311290370700334,-1.0422972306761351>, <0.10303700744140204,0.4026550926514324,-0.068351465260164>, 0.5 }
    cylinder {  m*<-2.6154498953795495,-4.736254647303032,-1.643425621608975>, <0.10303700744140204,0.4026550926514324,-0.068351465260164>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    