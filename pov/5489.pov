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
    sphere { m*<-0.8839740622132364,-0.16149227111486972,-1.404866082148683>, 1 }        
    sphere {  m*<0.2616271526432104,0.2849788961850331,8.519237655975635>, 1 }
    sphere {  m*<4.883630177422978,0.04431804339505063,-4.239934260752344>, 1 }
    sphere {  m*<-2.535148738158758,2.1673759645529374,-2.326813628914241>, 1}
    sphere { m*<-2.2673615171209267,-2.72031597785096,-2.1372673437516707>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2616271526432104,0.2849788961850331,8.519237655975635>, <-0.8839740622132364,-0.16149227111486972,-1.404866082148683>, 0.5 }
    cylinder { m*<4.883630177422978,0.04431804339505063,-4.239934260752344>, <-0.8839740622132364,-0.16149227111486972,-1.404866082148683>, 0.5}
    cylinder { m*<-2.535148738158758,2.1673759645529374,-2.326813628914241>, <-0.8839740622132364,-0.16149227111486972,-1.404866082148683>, 0.5 }
    cylinder {  m*<-2.2673615171209267,-2.72031597785096,-2.1372673437516707>, <-0.8839740622132364,-0.16149227111486972,-1.404866082148683>, 0.5}

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
    sphere { m*<-0.8839740622132364,-0.16149227111486972,-1.404866082148683>, 1 }        
    sphere {  m*<0.2616271526432104,0.2849788961850331,8.519237655975635>, 1 }
    sphere {  m*<4.883630177422978,0.04431804339505063,-4.239934260752344>, 1 }
    sphere {  m*<-2.535148738158758,2.1673759645529374,-2.326813628914241>, 1}
    sphere { m*<-2.2673615171209267,-2.72031597785096,-2.1372673437516707>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2616271526432104,0.2849788961850331,8.519237655975635>, <-0.8839740622132364,-0.16149227111486972,-1.404866082148683>, 0.5 }
    cylinder { m*<4.883630177422978,0.04431804339505063,-4.239934260752344>, <-0.8839740622132364,-0.16149227111486972,-1.404866082148683>, 0.5}
    cylinder { m*<-2.535148738158758,2.1673759645529374,-2.326813628914241>, <-0.8839740622132364,-0.16149227111486972,-1.404866082148683>, 0.5 }
    cylinder {  m*<-2.2673615171209267,-2.72031597785096,-2.1372673437516707>, <-0.8839740622132364,-0.16149227111486972,-1.404866082148683>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    