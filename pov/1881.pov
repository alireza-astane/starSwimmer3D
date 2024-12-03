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
    sphere { m*<1.150913616281553,1.6994416187513545e-20,0.697549695667242>, 1 }        
    sphere {  m*<1.3606309355957233,1.3650373046044172e-18,3.69021902531367>, 1 }
    sphere {  m*<4.660032342528796,7.153219233586036e-18,-0.827117594762987>, 1 }
    sphere {  m*<-3.784631798160837,8.164965809277259,-2.2980994736747604>, 1}
    sphere { m*<-3.784631798160837,-8.164965809277259,-2.298099473674764>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3606309355957233,1.3650373046044172e-18,3.69021902531367>, <1.150913616281553,1.6994416187513545e-20,0.697549695667242>, 0.5 }
    cylinder { m*<4.660032342528796,7.153219233586036e-18,-0.827117594762987>, <1.150913616281553,1.6994416187513545e-20,0.697549695667242>, 0.5}
    cylinder { m*<-3.784631798160837,8.164965809277259,-2.2980994736747604>, <1.150913616281553,1.6994416187513545e-20,0.697549695667242>, 0.5 }
    cylinder {  m*<-3.784631798160837,-8.164965809277259,-2.298099473674764>, <1.150913616281553,1.6994416187513545e-20,0.697549695667242>, 0.5}

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
    sphere { m*<1.150913616281553,1.6994416187513545e-20,0.697549695667242>, 1 }        
    sphere {  m*<1.3606309355957233,1.3650373046044172e-18,3.69021902531367>, 1 }
    sphere {  m*<4.660032342528796,7.153219233586036e-18,-0.827117594762987>, 1 }
    sphere {  m*<-3.784631798160837,8.164965809277259,-2.2980994736747604>, 1}
    sphere { m*<-3.784631798160837,-8.164965809277259,-2.298099473674764>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3606309355957233,1.3650373046044172e-18,3.69021902531367>, <1.150913616281553,1.6994416187513545e-20,0.697549695667242>, 0.5 }
    cylinder { m*<4.660032342528796,7.153219233586036e-18,-0.827117594762987>, <1.150913616281553,1.6994416187513545e-20,0.697549695667242>, 0.5}
    cylinder { m*<-3.784631798160837,8.164965809277259,-2.2980994736747604>, <1.150913616281553,1.6994416187513545e-20,0.697549695667242>, 0.5 }
    cylinder {  m*<-3.784631798160837,-8.164965809277259,-2.298099473674764>, <1.150913616281553,1.6994416187513545e-20,0.697549695667242>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    