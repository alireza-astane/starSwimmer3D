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
    sphere { m*<-2.0188340030642435e-18,3.927264007758726e-18,0.08526116968815657>, 1 }        
    sphere {  m*<-3.667063150884375e-18,5.194096292264825e-19,9.623261169688151>, 1 }
    sphere {  m*<9.428090415820634,1.3795649017930828e-18,-3.2480721636451775>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.2480721636451775>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.2480721636451775>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.667063150884375e-18,5.194096292264825e-19,9.623261169688151>, <-2.0188340030642435e-18,3.927264007758726e-18,0.08526116968815657>, 0.5 }
    cylinder { m*<9.428090415820634,1.3795649017930828e-18,-3.2480721636451775>, <-2.0188340030642435e-18,3.927264007758726e-18,0.08526116968815657>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.2480721636451775>, <-2.0188340030642435e-18,3.927264007758726e-18,0.08526116968815657>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.2480721636451775>, <-2.0188340030642435e-18,3.927264007758726e-18,0.08526116968815657>, 0.5}

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
    sphere { m*<-2.0188340030642435e-18,3.927264007758726e-18,0.08526116968815657>, 1 }        
    sphere {  m*<-3.667063150884375e-18,5.194096292264825e-19,9.623261169688151>, 1 }
    sphere {  m*<9.428090415820634,1.3795649017930828e-18,-3.2480721636451775>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.2480721636451775>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.2480721636451775>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.667063150884375e-18,5.194096292264825e-19,9.623261169688151>, <-2.0188340030642435e-18,3.927264007758726e-18,0.08526116968815657>, 0.5 }
    cylinder { m*<9.428090415820634,1.3795649017930828e-18,-3.2480721636451775>, <-2.0188340030642435e-18,3.927264007758726e-18,0.08526116968815657>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.2480721636451775>, <-2.0188340030642435e-18,3.927264007758726e-18,0.08526116968815657>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.2480721636451775>, <-2.0188340030642435e-18,3.927264007758726e-18,0.08526116968815657>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    