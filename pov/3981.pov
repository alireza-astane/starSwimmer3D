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
    sphere { m*<-0.1411208154076291,-0.05889026650514895,-0.20981496704891217>, 1 }        
    sphere {  m*<0.0996142893340623,0.06981981167517648,2.7777398040716372>, 1 }
    sphere {  m*<2.5935875785986324,0.04314370888122543,-1.439024492500097>, 1 }
    sphere {  m*<-1.7627361753005224,2.2695836779134537,-1.183760732464883>, 1}
    sphere { m*<-1.5518097360945902,-2.7255954204058135,-1.0271592254729414>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0996142893340623,0.06981981167517648,2.7777398040716372>, <-0.1411208154076291,-0.05889026650514895,-0.20981496704891217>, 0.5 }
    cylinder { m*<2.5935875785986324,0.04314370888122543,-1.439024492500097>, <-0.1411208154076291,-0.05889026650514895,-0.20981496704891217>, 0.5}
    cylinder { m*<-1.7627361753005224,2.2695836779134537,-1.183760732464883>, <-0.1411208154076291,-0.05889026650514895,-0.20981496704891217>, 0.5 }
    cylinder {  m*<-1.5518097360945902,-2.7255954204058135,-1.0271592254729414>, <-0.1411208154076291,-0.05889026650514895,-0.20981496704891217>, 0.5}

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
    sphere { m*<-0.1411208154076291,-0.05889026650514895,-0.20981496704891217>, 1 }        
    sphere {  m*<0.0996142893340623,0.06981981167517648,2.7777398040716372>, 1 }
    sphere {  m*<2.5935875785986324,0.04314370888122543,-1.439024492500097>, 1 }
    sphere {  m*<-1.7627361753005224,2.2695836779134537,-1.183760732464883>, 1}
    sphere { m*<-1.5518097360945902,-2.7255954204058135,-1.0271592254729414>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0996142893340623,0.06981981167517648,2.7777398040716372>, <-0.1411208154076291,-0.05889026650514895,-0.20981496704891217>, 0.5 }
    cylinder { m*<2.5935875785986324,0.04314370888122543,-1.439024492500097>, <-0.1411208154076291,-0.05889026650514895,-0.20981496704891217>, 0.5}
    cylinder { m*<-1.7627361753005224,2.2695836779134537,-1.183760732464883>, <-0.1411208154076291,-0.05889026650514895,-0.20981496704891217>, 0.5 }
    cylinder {  m*<-1.5518097360945902,-2.7255954204058135,-1.0271592254729414>, <-0.1411208154076291,-0.05889026650514895,-0.20981496704891217>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    