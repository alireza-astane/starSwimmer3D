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
    sphere { m*<-3.4933176136499346e-18,-5.354121265515446e-18,1.1214948753750478>, 1 }        
    sphere {  m*<-6.409118320799081e-18,-4.719744594320541e-18,4.618494875375088>, 1 }
    sphere {  m*<9.428090415820634,1.7354699151709153e-19,-2.211838457958285>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.211838457958285>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.211838457958285>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.409118320799081e-18,-4.719744594320541e-18,4.618494875375088>, <-3.4933176136499346e-18,-5.354121265515446e-18,1.1214948753750478>, 0.5 }
    cylinder { m*<9.428090415820634,1.7354699151709153e-19,-2.211838457958285>, <-3.4933176136499346e-18,-5.354121265515446e-18,1.1214948753750478>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.211838457958285>, <-3.4933176136499346e-18,-5.354121265515446e-18,1.1214948753750478>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.211838457958285>, <-3.4933176136499346e-18,-5.354121265515446e-18,1.1214948753750478>, 0.5}

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
    sphere { m*<-3.4933176136499346e-18,-5.354121265515446e-18,1.1214948753750478>, 1 }        
    sphere {  m*<-6.409118320799081e-18,-4.719744594320541e-18,4.618494875375088>, 1 }
    sphere {  m*<9.428090415820634,1.7354699151709153e-19,-2.211838457958285>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.211838457958285>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.211838457958285>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.409118320799081e-18,-4.719744594320541e-18,4.618494875375088>, <-3.4933176136499346e-18,-5.354121265515446e-18,1.1214948753750478>, 0.5 }
    cylinder { m*<9.428090415820634,1.7354699151709153e-19,-2.211838457958285>, <-3.4933176136499346e-18,-5.354121265515446e-18,1.1214948753750478>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.211838457958285>, <-3.4933176136499346e-18,-5.354121265515446e-18,1.1214948753750478>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.211838457958285>, <-3.4933176136499346e-18,-5.354121265515446e-18,1.1214948753750478>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    