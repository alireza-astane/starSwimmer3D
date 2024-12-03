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
    sphere { m*<-2.427118666651254e-18,2.293773194864612e-18,0.11097922946594115>, 1 }        
    sphere {  m*<-4.838972772390966e-18,1.743813624657109e-19,9.508979229465938>, 1 }
    sphere {  m*<9.428090415820634,-5.417489469428142e-19,-3.222354103867394>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.222354103867394>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.222354103867394>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.838972772390966e-18,1.743813624657109e-19,9.508979229465938>, <-2.427118666651254e-18,2.293773194864612e-18,0.11097922946594115>, 0.5 }
    cylinder { m*<9.428090415820634,-5.417489469428142e-19,-3.222354103867394>, <-2.427118666651254e-18,2.293773194864612e-18,0.11097922946594115>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.222354103867394>, <-2.427118666651254e-18,2.293773194864612e-18,0.11097922946594115>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.222354103867394>, <-2.427118666651254e-18,2.293773194864612e-18,0.11097922946594115>, 0.5}

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
    sphere { m*<-2.427118666651254e-18,2.293773194864612e-18,0.11097922946594115>, 1 }        
    sphere {  m*<-4.838972772390966e-18,1.743813624657109e-19,9.508979229465938>, 1 }
    sphere {  m*<9.428090415820634,-5.417489469428142e-19,-3.222354103867394>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.222354103867394>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.222354103867394>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.838972772390966e-18,1.743813624657109e-19,9.508979229465938>, <-2.427118666651254e-18,2.293773194864612e-18,0.11097922946594115>, 0.5 }
    cylinder { m*<9.428090415820634,-5.417489469428142e-19,-3.222354103867394>, <-2.427118666651254e-18,2.293773194864612e-18,0.11097922946594115>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.222354103867394>, <-2.427118666651254e-18,2.293773194864612e-18,0.11097922946594115>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.222354103867394>, <-2.427118666651254e-18,2.293773194864612e-18,0.11097922946594115>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    