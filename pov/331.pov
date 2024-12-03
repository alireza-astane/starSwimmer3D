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
    sphere { m*<-4.8234565931849165e-18,-2.587304719673395e-19,0.4223023578193517>, 1 }        
    sphere {  m*<-6.877276316487928e-18,-2.095823312972649e-18,8.098302357819367>, 1 }
    sphere {  m*<9.428090415820634,-2.8630909390816387e-18,-2.9110309755139805>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.9110309755139805>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.9110309755139805>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.877276316487928e-18,-2.095823312972649e-18,8.098302357819367>, <-4.8234565931849165e-18,-2.587304719673395e-19,0.4223023578193517>, 0.5 }
    cylinder { m*<9.428090415820634,-2.8630909390816387e-18,-2.9110309755139805>, <-4.8234565931849165e-18,-2.587304719673395e-19,0.4223023578193517>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.9110309755139805>, <-4.8234565931849165e-18,-2.587304719673395e-19,0.4223023578193517>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.9110309755139805>, <-4.8234565931849165e-18,-2.587304719673395e-19,0.4223023578193517>, 0.5}

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
    sphere { m*<-4.8234565931849165e-18,-2.587304719673395e-19,0.4223023578193517>, 1 }        
    sphere {  m*<-6.877276316487928e-18,-2.095823312972649e-18,8.098302357819367>, 1 }
    sphere {  m*<9.428090415820634,-2.8630909390816387e-18,-2.9110309755139805>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.9110309755139805>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.9110309755139805>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.877276316487928e-18,-2.095823312972649e-18,8.098302357819367>, <-4.8234565931849165e-18,-2.587304719673395e-19,0.4223023578193517>, 0.5 }
    cylinder { m*<9.428090415820634,-2.8630909390816387e-18,-2.9110309755139805>, <-4.8234565931849165e-18,-2.587304719673395e-19,0.4223023578193517>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.9110309755139805>, <-4.8234565931849165e-18,-2.587304719673395e-19,0.4223023578193517>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.9110309755139805>, <-4.8234565931849165e-18,-2.587304719673395e-19,0.4223023578193517>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    