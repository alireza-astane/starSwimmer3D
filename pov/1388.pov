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
    sphere { m*<0.534715758013909,-4.939850830615528e-18,0.9938560476857861>, 1 }        
    sphere {  m*<0.6136438048936669,-7.145101317139753e-19,3.992820281406285>, 1 }
    sphere {  m*<7.329296600874398,2.937746549050972e-18,-1.6116249565830383>, 1 }
    sphere {  m*<-4.2653997300740105,8.164965809277259,-2.2143029738381355>, 1}
    sphere { m*<-4.2653997300740105,-8.164965809277259,-2.214302973838139>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6136438048936669,-7.145101317139753e-19,3.992820281406285>, <0.534715758013909,-4.939850830615528e-18,0.9938560476857861>, 0.5 }
    cylinder { m*<7.329296600874398,2.937746549050972e-18,-1.6116249565830383>, <0.534715758013909,-4.939850830615528e-18,0.9938560476857861>, 0.5}
    cylinder { m*<-4.2653997300740105,8.164965809277259,-2.2143029738381355>, <0.534715758013909,-4.939850830615528e-18,0.9938560476857861>, 0.5 }
    cylinder {  m*<-4.2653997300740105,-8.164965809277259,-2.214302973838139>, <0.534715758013909,-4.939850830615528e-18,0.9938560476857861>, 0.5}

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
    sphere { m*<0.534715758013909,-4.939850830615528e-18,0.9938560476857861>, 1 }        
    sphere {  m*<0.6136438048936669,-7.145101317139753e-19,3.992820281406285>, 1 }
    sphere {  m*<7.329296600874398,2.937746549050972e-18,-1.6116249565830383>, 1 }
    sphere {  m*<-4.2653997300740105,8.164965809277259,-2.2143029738381355>, 1}
    sphere { m*<-4.2653997300740105,-8.164965809277259,-2.214302973838139>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6136438048936669,-7.145101317139753e-19,3.992820281406285>, <0.534715758013909,-4.939850830615528e-18,0.9938560476857861>, 0.5 }
    cylinder { m*<7.329296600874398,2.937746549050972e-18,-1.6116249565830383>, <0.534715758013909,-4.939850830615528e-18,0.9938560476857861>, 0.5}
    cylinder { m*<-4.2653997300740105,8.164965809277259,-2.2143029738381355>, <0.534715758013909,-4.939850830615528e-18,0.9938560476857861>, 0.5 }
    cylinder {  m*<-4.2653997300740105,-8.164965809277259,-2.214302973838139>, <0.534715758013909,-4.939850830615528e-18,0.9938560476857861>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    