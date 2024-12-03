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
    sphere { m*<-3.611165695761429e-18,-7.04182598494425e-20,0.5188390688352837>, 1 }        
    sphere {  m*<-3.3834548820414445e-18,-4.768898624032402e-18,7.648839068835304>, 1 }
    sphere {  m*<9.428090415820634,-2.8165552650006585e-18,-2.814494264498049>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.814494264498049>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.814494264498049>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.3834548820414445e-18,-4.768898624032402e-18,7.648839068835304>, <-3.611165695761429e-18,-7.04182598494425e-20,0.5188390688352837>, 0.5 }
    cylinder { m*<9.428090415820634,-2.8165552650006585e-18,-2.814494264498049>, <-3.611165695761429e-18,-7.04182598494425e-20,0.5188390688352837>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.814494264498049>, <-3.611165695761429e-18,-7.04182598494425e-20,0.5188390688352837>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.814494264498049>, <-3.611165695761429e-18,-7.04182598494425e-20,0.5188390688352837>, 0.5}

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
    sphere { m*<-3.611165695761429e-18,-7.04182598494425e-20,0.5188390688352837>, 1 }        
    sphere {  m*<-3.3834548820414445e-18,-4.768898624032402e-18,7.648839068835304>, 1 }
    sphere {  m*<9.428090415820634,-2.8165552650006585e-18,-2.814494264498049>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.814494264498049>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.814494264498049>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.3834548820414445e-18,-4.768898624032402e-18,7.648839068835304>, <-3.611165695761429e-18,-7.04182598494425e-20,0.5188390688352837>, 0.5 }
    cylinder { m*<9.428090415820634,-2.8165552650006585e-18,-2.814494264498049>, <-3.611165695761429e-18,-7.04182598494425e-20,0.5188390688352837>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.814494264498049>, <-3.611165695761429e-18,-7.04182598494425e-20,0.5188390688352837>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.814494264498049>, <-3.611165695761429e-18,-7.04182598494425e-20,0.5188390688352837>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    